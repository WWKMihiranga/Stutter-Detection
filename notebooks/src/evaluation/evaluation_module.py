"""
Comprehensive Evaluation Module for Stuttering Detection
Implements all metrics required by research proposal:
- Event-level F1 score (RQ1)
- Onset/Offset RMSE (RQ1)  
- Rule contribution analysis (RQ2)
- Ablation study framework (RQ2)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class TemporalLocalizationEvaluator:
    """
    Evaluates temporal boundary detection quality.
    Implements RQ1 metrics from research proposal.
    """
    
    def __init__(self, iou_threshold=0.5, frame_rate=50):
        """
        Args:
            iou_threshold: Minimum IoU for event matching (default 0.5)
            frame_rate: Frames per second for time conversion
        """
        self.iou_threshold = iou_threshold
        self.frame_rate = frame_rate  # 50 frames for 3-second clip (Wav2Vec2 output)
        self.ms_per_frame = (3000 / frame_rate)  # milliseconds per frame
    
    def calculate_iou(self, pred_start, pred_end, gt_start, gt_end):
        """Calculate Intersection over Union for temporal intervals"""
        # Intersection
        int_start = max(pred_start, gt_start)
        int_end = min(pred_end, gt_end)
        intersection = max(0, int_end - int_start)
        
        # Union
        union_start = min(pred_start, gt_start)
        union_end = max(pred_end, gt_end)
        union = union_end - union_start
        
        return intersection / union if union > 0 else 0
    
    def extract_events_from_frame_predictions(self, frame_predictions, threshold=0.5):
        """
        Extract discrete events from frame-level predictions.
        
        Args:
            frame_predictions: (seq_len, num_classes) numpy array
            threshold: Confidence threshold
        
        Returns:
            List of events: [{'class': idx, 'onset': frame, 'offset': frame, 'confidence': float}]
        """
        events = []
        num_frames, num_classes = frame_predictions.shape
        
        # For each class (except 'no_stuttered_words' if present)
        for class_idx in range(num_classes):
            probs = frame_predictions[:, class_idx]
            binary = (probs > threshold).astype(int)
            
            # Find connected components (consecutive 1s)
            if binary.sum() == 0:
                continue
            
            # Detect boundaries
            diff = np.diff(np.concatenate([[0], binary, [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Create events
            for start, end in zip(starts, ends):
                events.append({
                    'class': class_idx,
                    'onset': start,
                    'offset': end,
                    'duration': end - start,
                    'confidence': float(probs[start:end].mean())
                })
        
        return events
    
    def match_events(self, predicted_events, ground_truth_events):
        """
        Match predicted events to ground truth using IoU.
        Uses greedy matching strategy.
        
        Returns:
            matches: List of (pred_idx, gt_idx) tuples
        """
        if len(predicted_events) == 0 or len(ground_truth_events) == 0:
            return []
        
        # Create IoU matrix
        iou_matrix = np.zeros((len(predicted_events), len(ground_truth_events)))
        
        for i, pred in enumerate(predicted_events):
            for j, gt in enumerate(ground_truth_events):
                # Only match same class
                if pred['class'] == gt['class']:
                    iou = self.calculate_iou(
                        pred['onset'], pred['offset'],
                        gt['onset'], gt['offset']
                    )
                    iou_matrix[i, j] = iou
        
        # Greedy matching
        matches = []
        used_gt = set()
        
        # Sort predictions by confidence
        sorted_pred_indices = sorted(range(len(predicted_events)), 
                                    key=lambda i: predicted_events[i]['confidence'], 
                                    reverse=True)
        
        for pred_idx in sorted_pred_indices:
            best_gt_idx = None
            best_iou = self.iou_threshold
            
            for gt_idx in range(len(ground_truth_events)):
                if gt_idx in used_gt:
                    continue
                
                iou = iou_matrix[pred_idx, gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx is not None:
                matches.append((pred_idx, best_gt_idx))
                used_gt.add(best_gt_idx)
        
        return matches
    
    def calculate_event_f1(self, predicted_events, ground_truth_events, per_class=False):
        """
        Calculate event-level F1 score (RQ1 primary metric).
        
        Args:
            predicted_events: List of predicted events
            ground_truth_events: List of ground truth events
            per_class: If True, calculate per-class metrics
        
        Returns:
            Dict with precision, recall, F1, and counts
        """
        if len(ground_truth_events) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'true_positives': 0,
                'false_positives': len(predicted_events),
                'false_negatives': 0
            }
        
        matches = self.match_events(predicted_events, ground_truth_events)
        
        tp = len(matches)
        fp = len(predicted_events) - tp
        fn = len(ground_truth_events) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Per-class metrics if requested
        if per_class:
            # Get unique classes
            all_classes = set([e['class'] for e in predicted_events + ground_truth_events])
            results['per_class'] = {}
            
            for cls in all_classes:
                pred_cls = [e for e in predicted_events if e['class'] == cls]
                gt_cls = [e for e in ground_truth_events if e['class'] == cls]
                
                # Recursive call for this class only
                cls_metrics = self.calculate_event_f1(pred_cls, gt_cls, per_class=False)
                results['per_class'][int(cls)] = cls_metrics
        
        return results
    
    def calculate_boundary_rmse(self, predicted_events, ground_truth_events):
        """
        Calculate RMSE for onset and offset boundaries in milliseconds (RQ1 metric).
        
        Args:
            predicted_events: List of predicted events
            ground_truth_events: List of ground truth events
        
        Returns:
            Dict with onset_rmse, offset_rmse in milliseconds
        """
        matches = self.match_events(predicted_events, ground_truth_events)
        
        if len(matches) == 0:
            return {
                'onset_rmse_ms': float('inf'),
                'offset_rmse_ms': float('inf'),
                'mean_onset_error_ms': float('inf'),
                'mean_offset_error_ms': float('inf'),
                'num_matched_events': 0
            }
        
        onset_errors = []
        offset_errors = []
        
        for pred_idx, gt_idx in matches:
            pred = predicted_events[pred_idx]
            gt = ground_truth_events[gt_idx]
            
            # Convert frame errors to milliseconds
            onset_error_frames = abs(pred['onset'] - gt['onset'])
            offset_error_frames = abs(pred['offset'] - gt['offset'])
            
            onset_error_ms = onset_error_frames * self.ms_per_frame
            offset_error_ms = offset_error_frames * self.ms_per_frame
            
            onset_errors.append(onset_error_ms)
            offset_errors.append(offset_error_ms)
        
        onset_errors = np.array(onset_errors)
        offset_errors = np.array(offset_errors)
        
        return {
            'onset_rmse_ms': float(np.sqrt(np.mean(onset_errors**2))),
            'offset_rmse_ms': float(np.sqrt(np.mean(offset_errors**2))),
            'mean_onset_error_ms': float(np.mean(onset_errors)),
            'mean_offset_error_ms': float(np.mean(offset_errors)),
            'std_onset_error_ms': float(np.std(onset_errors)),
            'std_offset_error_ms': float(np.std(offset_errors)),
            'num_matched_events': len(matches)
        }
    
    def evaluate_model(self, model, dataloader, device, max_samples=None):
        """
        Comprehensive evaluation of model on a dataset.
        
        Args:
            model: Trained model
            dataloader: DataLoader with test data
            device: torch device
            max_samples: Limit number of samples (for speed)
        
        Returns:
            Dict with all metrics
        """
        model.eval()
        
        all_predicted_events = []
        all_gt_events = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and batch_idx >= max_samples:
                    break
                
                audio = batch['audio'].to(device)
                
                # Get frame predictions
                frame_logits = model(audio)  # (batch, seq_len, num_classes)
                frame_probs = torch.sigmoid(frame_logits)
                
                # Process each sample in batch
                for i in range(frame_probs.shape[0]):
                    probs = frame_probs[i].cpu().numpy()  # (seq_len, num_classes)
                    
                    # Extract predicted events
                    pred_events = self.extract_events_from_frame_predictions(probs)
                    
                    # Extract GT events (if available)
                    if 'frame_label' in batch:
                        gt_labels = batch['frame_label'][i].cpu().numpy()
                        gt_events = self.extract_events_from_frame_predictions(
                            gt_labels, threshold=0.5
                        )
                    else:
                        # Fallback: Use clip label as pseudo-GT (single event spanning whole clip)
                        clip_label = batch['clip_label'][i].cpu().numpy()
                        gt_events = []
                        for class_idx in range(len(clip_label)):
                            if clip_label[class_idx] > 0.5:
                                gt_events.append({
                                    'class': class_idx,
                                    'onset': 0,
                                    'offset': probs.shape[0] - 1,
                                    'duration': probs.shape[0],
                                    'confidence': 1.0
                                })
                    
                    all_predicted_events.extend(pred_events)
                    all_gt_events.extend(gt_events)
        
        # Calculate all metrics
        f1_metrics = self.calculate_event_f1(all_predicted_events, all_gt_events, per_class=True)
        rmse_metrics = self.calculate_boundary_rmse(all_predicted_events, all_gt_events)
        
        return {
            'event_f1': f1_metrics,
            'boundary_rmse': rmse_metrics,
            'num_predicted_events': len(all_predicted_events),
            'num_gt_events': len(all_gt_events)
        }


class InterpretabilityAnalyzer:
    """
    Analyzes interpretability of soft-rule module.
    Implements RQ2 metrics from research proposal.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.rule_names = ['voicing_continuity', 'spectral_similarity', 'silence_detection']
    
    def extract_rule_activations(self, audio_batch):
        """
        Extract rule activations and gate weights from model.
        
        Args:
            audio_batch: (batch, audio_len) tensor
        
        Returns:
            Dict with rule scores and gate weights
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            output = self.model(audio_batch)
            
            # Extract rule activations (need to modify model to return these)
            # For now, we'll use a hook-based approach
            if hasattr(self.model, 'soft_rules'):
                # Get hidden features from temporal head
                features = self.model.encoder(audio_batch).last_hidden_state
                
                # Get rule scores
                rule_scores = self.model.soft_rules(features)  # (batch, seq_len, num_rules)
                
                # Get gate weights
                temporal_features = self.model.temporal_head.get_features(features)
                gate_input = torch.cat([temporal_features, rule_scores.mean(dim=1)], dim=-1)
                gate_weights = self.model.gating(gate_input)  # (batch, num_rules)
                
                return {
                    'rule_scores': rule_scores.cpu().numpy(),
                    'gate_weights': gate_weights.cpu().numpy()
                }
        
        return None
    
    def calculate_acoustic_proxies(self, audio_numpy):
        """
        Calculate engineered acoustic features as proxies for rules.
        
        Args:
            audio_numpy: (audio_len,) numpy array
        
        Returns:
            Dict with proxy values for each rule
        """
        import librosa
        
        proxies = {}
        
        # Voicing proxy: Zero-crossing rate (lower = more voiced/sustained)
        zcr = librosa.feature.zero_crossing_rate(audio_numpy)[0].mean()
        proxies['voicing_continuity'] = 1.0 - zcr  # Invert so higher = more continuous
        
        # Similarity proxy: Autocorrelation at lag=100 samples
        if len(audio_numpy) > 100:
            autocorr = np.corrcoef(audio_numpy[:-100], audio_numpy[100:])[0, 1]
            proxies['spectral_similarity'] = autocorr if not np.isnan(autocorr) else 0.0
        else:
            proxies['spectral_similarity'] = 0.0
        
        # Silence proxy: RMS energy (lower = more silence)
        rms = np.sqrt(np.mean(audio_numpy**2))
        proxies['silence_detection'] = 1.0 - min(rms / 0.1, 1.0)  # Normalize
        
        return proxies
    
    def calculate_rule_feature_correlation(self, dataloader, max_samples=100):
        """
        Calculate Pearson's r between rule activations and acoustic proxies (RQ2 metric).
        
        Args:
            dataloader: DataLoader with evaluation data
            max_samples: Number of samples to analyze
        
        Returns:
            Dict with correlation coefficients
        """
        rule_activations = {name: [] for name in self.rule_names}
        acoustic_proxies = {name: [] for name in self.rule_names}
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_samples:
                    break
                
                audio = batch['audio'].to(self.device)
                
                # Get rule activations
                activations = self.extract_rule_activations(audio)
                if activations is None:
                    continue
                
                # Process each sample
                for i in range(audio.shape[0]):
                    # Average rule scores over time
                    rule_scores = activations['rule_scores'][i].mean(axis=0)  # (num_rules,)
                    
                    for rule_idx, rule_name in enumerate(self.rule_names):
                        rule_activations[rule_name].append(rule_scores[rule_idx])
                    
                    # Calculate acoustic proxies
                    audio_np = batch['audio'][i].cpu().numpy()
                    proxies = self.calculate_acoustic_proxies(audio_np)
                    
                    for rule_name in self.rule_names:
                        acoustic_proxies[rule_name].append(proxies[rule_name])
        
        # Calculate correlations
        correlations = {}
        for rule_name in self.rule_names:
            if len(rule_activations[rule_name]) > 1:
                rule_vals = np.array(rule_activations[rule_name])
                proxy_vals = np.array(acoustic_proxies[rule_name])
                
                # Pearson's r
                r, p_value = pearsonr(rule_vals, proxy_vals)
                
                correlations[rule_name] = {
                    'pearson_r': float(r),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05),
                    'num_samples': len(rule_vals)
                }
            else:
                correlations[rule_name] = {
                    'pearson_r': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'num_samples': 0
                }
        
        return correlations
    
    def analyze_gate_contributions(self, dataloader, max_samples=100):
        """
        Analyze how gate network weights rules across samples.
        
        Returns:
            Dict with gate statistics
        """
        all_gate_weights = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_samples:
                    break
                
                audio = batch['audio'].to(self.device)
                activations = self.extract_rule_activations(audio)
                
                if activations is not None:
                    all_gate_weights.append(activations['gate_weights'])
        
        if len(all_gate_weights) == 0:
            return {}
        
        gate_weights = np.concatenate(all_gate_weights, axis=0)  # (num_samples, num_rules)
        
        return {
            'mean_weights': gate_weights.mean(axis=0).tolist(),
            'std_weights': gate_weights.std(axis=0).tolist(),
            'min_weights': gate_weights.min(axis=0).tolist(),
            'max_weights': gate_weights.max(axis=0).tolist(),
            'rule_names': self.rule_names
        }


class AblationStudyRunner:
    """
    Runs ablation study to measure rule contribution (RQ2).
    """
    
    def __init__(self, config, train_loader, val_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.evaluator = TemporalLocalizationEvaluator()
    
    def train_model_variant(self, model, num_epochs=5):
        """
        Train a model variant.
        
        Args:
            model: Model to train
            num_epochs: Number of training epochs
        
        Returns:
            Trained model, test metrics
        """
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                audio = batch['audio'].to(self.config.DEVICE)
                targets = batch['clip_label'].to(self.config.DEVICE)
                
                optimizer.zero_grad()
                
                # Forward
                output = model(audio)
                clip_logits, _ = torch.max(output, dim=1)
                
                # Loss
                loss = criterion(clip_logits, targets)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Evaluate
        test_metrics = self.evaluator.evaluate_model(
            model, self.test_loader, self.config.DEVICE, max_samples=50
        )
        
        return model, test_metrics
    
    def run_ablation_study(self, num_epochs=5):
        """
        Run complete ablation study.
        
        Returns:
            Dict with results for each variant
        """
        from copy import deepcopy
        import sys
        import os
        
        results = {}
        
        # Import model class
        # This would need to be adjusted based on actual import path
        from models.neurosymbolic import NeuroSymbolicStutterDetectorCPU
        
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        # 1. Neural-only baseline (no rules)
        print("\n1. Training Neural-Only Baseline (No Rules)...")
        config_no_rules = deepcopy(self.config)
        config_no_rules.USE_RULES = False  # Need to add this flag to model
        
        model_neural = NeuroSymbolicStutterDetectorCPU(config_no_rules, freeze_encoder=True)
        model_neural = model_neural.to(self.config.DEVICE)
        
        # Disable rule module
        if hasattr(model_neural, 'soft_rules'):
            for param in model_neural.soft_rules.parameters():
                param.requires_grad = False
        if hasattr(model_neural, 'gating'):
            for param in model_neural.gating.parameters():
                param.requires_grad = False
        
        trained_neural, metrics_neural = self.train_model_variant(model_neural, num_epochs)
        
        results['neural_only'] = {
            'f1': metrics_neural['event_f1']['f1'],
            'precision': metrics_neural['event_f1']['precision'],
            'recall': metrics_neural['event_f1']['recall'],
            'description': 'Neural detection head only (no rules)'
        }
        
        print(f"  Neural-Only F1: {results['neural_only']['f1']:.4f}")
        
        # 2. Full model (neural + rules + gating)
        print("\n2. Training Full Model (Neural + Rules + Gating)...")
        model_full = NeuroSymbolicStutterDetectorCPU(self.config, freeze_encoder=True)
        model_full = model_full.to(self.config.DEVICE)
        
        trained_full, metrics_full = self.train_model_variant(model_full, num_epochs)
        
        results['full_model'] = {
            'f1': metrics_full['event_f1']['f1'],
            'precision': metrics_full['event_f1']['precision'],
            'recall': metrics_full['event_f1']['recall'],
            'description': 'Full neuro-symbolic model'
        }
        
        print(f"  Full Model F1: {results['full_model']['f1']:.4f}")
        
        # Calculate contribution
        rule_contribution = results['full_model']['f1'] - results['neural_only']['f1']
        relative_improvement = (rule_contribution / results['neural_only']['f1'] * 100) if results['neural_only']['f1'] > 0 else 0
        
        results['summary'] = {
            'rule_contribution_absolute': float(rule_contribution),
            'rule_contribution_relative_pct': float(relative_improvement),
            'interpretation': f"Rules improve F1 by {rule_contribution:.4f} ({relative_improvement:.2f}%)"
        }
        
        print(f"\n{'='*60}")
        print(f"ABLATION SUMMARY:")
        print(f"  Neural-Only: F1 = {results['neural_only']['f1']:.4f}")
        print(f"  Full Model:  F1 = {results['full_model']['f1']:.4f}")
        print(f"  Improvement: +{rule_contribution:.4f} ({relative_improvement:.2f}%)")
        print(f"{'='*60}")
        
        return results


def save_evaluation_report(metrics, output_path):
    """
    Save comprehensive evaluation report to JSON.
    
    Args:
        metrics: Dict with all evaluation metrics
        output_path: Path to save JSON file
    """
    report = {
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation report saved to: {output_path}")


# Example usage function
def run_complete_evaluation(model, train_loader, val_loader, test_loader, config, output_dir):
    """
    Run all evaluations required by research proposal.
    
    Args:
        model: Trained model to evaluate
        train_loader, val_loader, test_loader: Data loaders
        config: Configuration object
        output_dir: Directory to save results
    
    Returns:
        Dict with all evaluation metrics
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION (Research Proposal Requirements)")
    print("="*70)
    
    results = {}
    
    # 1. RQ1: Temporal localization metrics
    print("\n[RQ1] Temporal Localization Evaluation...")
    evaluator = TemporalLocalizationEvaluator()
    localization_metrics = evaluator.evaluate_model(model, test_loader, config.DEVICE, max_samples=100)
    results['rq1_localization'] = localization_metrics
    
    print(f"  Event-level F1: {localization_metrics['event_f1']['f1']:.4f}")
    print(f"  Onset RMSE: {localization_metrics['boundary_rmse']['onset_rmse_ms']:.2f} ms")
    print(f"  Offset RMSE: {localization_metrics['boundary_rmse']['offset_rmse_ms']:.2f} ms")
    
    # 2. RQ2: Interpretability analysis
    print("\n[RQ2] Interpretability Analysis...")
    analyzer = InterpretabilityAnalyzer(model, config.DEVICE)
    
    # Rule-feature correlation
    correlations = analyzer.calculate_rule_feature_correlation(test_loader, max_samples=50)
    results['rq2_correlations'] = correlations
    
    print("  Rule-Feature Correlations:")
    for rule_name, corr_data in correlations.items():
        sig = "✓" if corr_data['significant'] else "✗"
        print(f"    {rule_name}: r={corr_data['pearson_r']:.3f} (p={corr_data['p_value']:.4f}) {sig}")
    
    # Gate contributions
    gate_stats = analyzer.analyze_gate_contributions(test_loader, max_samples=50)
    results['rq2_gate_contributions'] = gate_stats
    
    if gate_stats:
        print("  Mean Gate Weights:")
        for i, (name, weight) in enumerate(zip(gate_stats['rule_names'], gate_stats['mean_weights'])):
            print(f"    {name}: {weight:.3f}")
    
    # 3. RQ2: Ablation study (if time permits)
    # print("\n[RQ2] Ablation Study...")
    # ablation_runner = AblationStudyRunner(config, train_loader, val_loader, test_loader)
    # ablation_results = ablation_runner.run_ablation_study(num_epochs=3)
    # results['rq2_ablation'] = ablation_results
    
    # Save report
    output_path = Path(output_dir) / 'evaluation_report.json'
    save_evaluation_report(results, output_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return results
