"""
Improved Pseudo-Label Generation for Stage 2 Training
Addresses the low frame coverage issue (was 0.13% → 1.73%)
Uses adaptive thresholding and confidence-based strategies
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.ndimage import label as scipy_label


class ImprovedPseudoLabelGenerator:
    """
    Generates high-quality frame-level pseudo-labels from clip-level predictions.
    Fixes the sparse pseudo-label problem from original implementation.
    """
    
    def __init__(self, 
                 model, 
                 base_confidence=0.25,  # Lower threshold for better coverage
                 min_event_length=2,     # Minimum 2 frames (shorter events OK)
                 use_adaptive_threshold=True,
                 use_smoothing=True):
        """
        Args:
            model: Trained Stage 1 model
            base_confidence: Base threshold for pseudo-labels
            min_event_length: Minimum frames for valid event
            use_adaptive_threshold: Adapt threshold per class
            use_smoothing: Apply temporal smoothing
        """
        self.model = model
        self.base_confidence = base_confidence
        self.min_event_length = min_event_length
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_smoothing = use_smoothing
        
        # Will be calculated adaptively
        self.class_thresholds = {}
        
        print(f"ImprovedPseudoLabelGenerator initialized:")
        print(f"  Base confidence: {base_confidence}")
        print(f"  Min event length: {min_event_length}")
        print(f"  Adaptive threshold: {use_adaptive_threshold}")
        print(f"  Smoothing: {use_smoothing}")
    
    def calibrate_thresholds(self, dataloader, num_samples=100):
        """
        Calibrate class-specific thresholds based on prediction distribution.
        This helps address class imbalance.
        
        Args:
            dataloader: Training data loader
            num_samples: Number of samples to use for calibration
        """
        print("\nCalibrating class-specific thresholds...")
        
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                
                audio = batch['audio'].to(next(self.model.parameters()).device)
                targets = batch['clip_label']
                
                # Get predictions
                output = self.model(audio)
                frame_probs = torch.sigmoid(output)
                
                # Max pool to get clip predictions
                clip_probs, _ = torch.max(frame_probs, dim=1)
                
                all_probs.append(clip_probs.cpu().numpy())
                all_labels.append(targets.numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)  # (num_samples, num_classes)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate threshold for each class
        num_classes = all_probs.shape[1]
        
        for class_idx in range(num_classes):
            class_probs = all_probs[:, class_idx]
            class_labels = all_labels[:, class_idx]
            
            # Get probabilities for positive samples
            pos_probs = class_probs[class_labels > 0.5]
            
            if len(pos_probs) > 0:
                # Use 25th percentile of positive samples
                threshold = np.percentile(pos_probs, 25)
                # Ensure it's not too low
                threshold = max(threshold, self.base_confidence)
                # Ensure it's not too high
                threshold = min(threshold, 0.7)
            else:
                threshold = self.base_confidence
            
            self.class_thresholds[class_idx] = threshold
        
        print("  Class thresholds:")
        for class_idx, threshold in self.class_thresholds.items():
            print(f"    Class {class_idx}: {threshold:.3f}")
    
    def smooth_predictions(self, frame_probs, kernel_size=3):
        """
        Apply temporal smoothing to reduce noise.
        
        Args:
            frame_probs: (seq_len, num_classes) array
            kernel_size: Size of smoothing window
        
        Returns:
            Smoothed probabilities
        """
        from scipy.ndimage import uniform_filter1d
        
        smoothed = np.zeros_like(frame_probs)
        for class_idx in range(frame_probs.shape[1]):
            smoothed[:, class_idx] = uniform_filter1d(
                frame_probs[:, class_idx], 
                size=kernel_size, 
                mode='nearest'
            )
        
        return smoothed
    
    def extract_events_with_confidence(self, frame_probs, clip_label):
        """
        Extract events from frame predictions with quality filtering.
        
        Args:
            frame_probs: (seq_len, num_classes) probabilities
            clip_label: (num_classes,) clip-level labels
        
        Returns:
            frame_labels, events, metadata
        """
        seq_len, num_classes = frame_probs.shape
        frame_labels = np.zeros_like(frame_probs)
        events = []
        
        # Only process classes that are present in clip label
        active_classes = np.where(clip_label > 0.5)[0]
        
        for class_idx in active_classes:
            # Get threshold for this class
            if self.use_adaptive_threshold and class_idx in self.class_thresholds:
                threshold = self.class_thresholds[class_idx]
            else:
                threshold = self.base_confidence
            
            # Get class probabilities
            class_probs = frame_probs[:, class_idx]
            
            # Apply threshold
            binary = (class_probs > threshold).astype(int)
            
            if binary.sum() == 0:
                continue
            
            # Find connected components
            labeled_array, num_features = scipy_label(binary)
            
            for event_id in range(1, num_features + 1):
                event_mask = (labeled_array == event_id)
                event_indices = np.where(event_mask)[0]
                
                if len(event_indices) < self.min_event_length:
                    continue
                
                start_idx = event_indices[0]
                end_idx = event_indices[-1] + 1
                
                # Calculate event confidence
                event_probs = class_probs[start_idx:end_idx]
                confidence = float(event_probs.mean())
                
                # Only keep high-confidence events
                if confidence > threshold:
                    # Mark in frame labels
                    frame_labels[start_idx:end_idx, class_idx] = 1.0
                    
                    events.append({
                        'class': int(class_idx),
                        'start': int(start_idx),
                        'end': int(end_idx),
                        'duration': int(end_idx - start_idx),
                        'confidence': confidence
                    })
        
        metadata = {
            'num_events': len(events),
            'frame_coverage': float((frame_labels > 0).sum() / frame_labels.size * 100),
            'active_classes': active_classes.tolist()
        }
        
        return frame_labels, events, metadata
    
    def generate_pseudo_labels(self, dataloader, device, max_samples=None):
        """
        Generate pseudo-labels for entire dataset.
        
        Args:
            dataloader: DataLoader for training data
            device: torch device
            max_samples: Limit number of samples (for testing)
        
        Returns:
            pseudo_labels_dict, statistics
        """
        print("\nGenerating pseudo-labels...")
        
        self.model.eval()
        pseudo_labels_dict = {}
        
        stats = {
            'total_samples': 0,
            'samples_with_events': 0,
            'total_events': 0,
            'total_frames': 0,
            'frames_with_labels': 0,
            'per_class_events': {},
            'avg_confidence': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and batch_idx >= max_samples:
                    break
                
                audio = batch['audio'].to(device)
                targets = batch['clip_label'].cpu().numpy()
                file_paths = batch.get('file_path', [f'sample_{batch_idx}'])
                
                # Get frame predictions
                output = self.model(audio)
                frame_probs = torch.sigmoid(output).cpu().numpy()
                
                # Process each sample in batch
                for i in range(frame_probs.shape[0]):
                    probs = frame_probs[i]  # (seq_len, num_classes)
                    clip_label = targets[i]
                    
                    # Apply smoothing if enabled
                    if self.use_smoothing:
                        probs = self.smooth_predictions(probs)
                    
                    # Extract events
                    frame_labels, events, metadata = self.extract_events_with_confidence(
                        probs, clip_label
                    )
                    
                    # Store pseudo-labels
                    file_path = file_paths[i] if isinstance(file_paths, list) else f'sample_{batch_idx}_{i}'
                    
                    pseudo_labels_dict[file_path] = {
                        'frame_labels': frame_labels,
                        'frame_probs': probs,
                        'clip_label': clip_label,
                        'events': events,
                        'metadata': metadata
                    }
                    
                    # Update statistics
                    stats['total_samples'] += 1
                    if len(events) > 0:
                        stats['samples_with_events'] += 1
                    stats['total_events'] += len(events)
                    stats['total_frames'] += frame_labels.size
                    stats['frames_with_labels'] += (frame_labels > 0).sum()
                    
                    for event in events:
                        class_idx = event['class']
                        if class_idx not in stats['per_class_events']:
                            stats['per_class_events'][class_idx] = 0
                        stats['per_class_events'][class_idx] += 1
                        stats['avg_confidence'].append(event['confidence'])
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
        
        # Calculate final statistics
        if stats['total_samples'] > 0:
            stats['samples_with_events_pct'] = stats['samples_with_events'] / stats['total_samples'] * 100
            stats['frame_coverage_pct'] = stats['frames_with_labels'] / stats['total_frames'] * 100
            stats['avg_events_per_sample'] = stats['total_events'] / stats['total_samples']
        
        if stats['avg_confidence']:
            stats['mean_confidence'] = float(np.mean(stats['avg_confidence']))
            stats['std_confidence'] = float(np.std(stats['avg_confidence']))
        
        print(f"\nPseudo-label generation complete!")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Samples with events: {stats['samples_with_events']} ({stats.get('samples_with_events_pct', 0):.1f}%)")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Frame coverage: {stats.get('frame_coverage_pct', 0):.2f}%")
        print(f"  Avg events/sample: {stats.get('avg_events_per_sample', 0):.3f}")
        
        return pseudo_labels_dict, stats
    
    def quality_check(self, pseudo_labels_dict, idx2label):
        """
        Perform quality check on generated pseudo-labels.
        
        Args:
            pseudo_labels_dict: Generated pseudo-labels
            idx2label: Class index to name mapping
        
        Returns:
            Quality report
        """
        print("\nQuality Check:")
        
        clip_accuracy = []
        
        for file_path, data in list(pseudo_labels_dict.items())[:100]:  # Check first 100
            clip_label = data['clip_label']
            frame_labels = data['frame_labels']
            
            # Check if predicted classes match clip label
            predicted_classes = np.where(frame_labels.sum(axis=0) > 0)[0]
            true_classes = np.where(clip_label > 0.5)[0]
            
            # Calculate accuracy
            if len(true_classes) > 0:
                correct = len(set(predicted_classes) & set(true_classes))
                accuracy = correct / len(true_classes)
                clip_accuracy.append(accuracy)
        
        if clip_accuracy:
            mean_accuracy = np.mean(clip_accuracy)
            print(f"  Clip-level class accuracy: {mean_accuracy:.3f}")
            print(f"  (How well pseudo-labels match clip labels)")
        
        # Check event distribution
        print(f"\n  Event distribution:")
        all_events = []
        for data in pseudo_labels_dict.values():
            all_events.extend(data['events'])
        
        if all_events:
            event_classes = [e['class'] for e in all_events]
            for class_idx in set(event_classes):
                count = event_classes.count(class_idx)
                class_name = idx2label.get(str(class_idx), f'class_{class_idx}')
                print(f"    {class_name}: {count} events")
        
        return {
            'clip_accuracy': float(np.mean(clip_accuracy)) if clip_accuracy else 0.0,
            'num_samples_checked': len(clip_accuracy)
        }


def create_improved_pseudo_labels(stage1_model, train_loader, device, config, idx2label):
    """
    Complete pipeline for improved pseudo-label generation.
    
    Args:
        stage1_model: Trained Stage 1 model
        train_loader: Training data loader
        device: torch device
        config: Configuration object
        idx2label: Class index to name mapping
    
    Returns:
        pseudo_labels_dict, statistics
    """
    print("\n" + "="*70)
    print("IMPROVED PSEUDO-LABEL GENERATION")
    print("="*70)
    
    # Create generator
    generator = ImprovedPseudoLabelGenerator(
        model=stage1_model,
        base_confidence=0.25,
        min_event_length=2,
        use_adaptive_threshold=True,
        use_smoothing=True
    )
    
    # Calibrate thresholds
    generator.calibrate_thresholds(train_loader, num_samples=100)
    
    # Generate pseudo-labels
    pseudo_labels_dict, stats = generator.generate_pseudo_labels(
        train_loader, 
        device,
        max_samples=None  # Process all samples
    )
    
    # Quality check
    quality_report = generator.quality_check(pseudo_labels_dict, idx2label)
    stats['quality'] = quality_report
    
    print("\n" + "="*70)
    print("PSEUDO-LABEL GENERATION COMPLETE")
    print("="*70)
    
    return pseudo_labels_dict, stats
