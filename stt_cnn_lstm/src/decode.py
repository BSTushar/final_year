from typing import List, Tuple

import torch


def ctc_greedy_decode(log_probs: torch.Tensor) -> List[List[int]]:
    """Greedy CTC decoding (fast but less accurate)."""
    probs = log_probs.detach().cpu().argmax(dim=-1)
    T, B = probs.shape
    results: List[List[int]] = []
    blank_id = 0
    for b in range(B):
        prev = blank_id
        seq: List[int] = []
        for t in range(T):
            p = int(probs[t, b])
            if p != blank_id and p != prev:
                seq.append(p)
            prev = p
        results.append(seq)
    return results


def ctc_beam_decode(log_probs: torch.Tensor, beam_width: int = 5) -> List[List[int]]:
    """
    Beam search decoding for CTC (better accuracy than greedy).
    
    This is a simplified beam search implementation that works well for CTC.
    Inspired by the referenced project's approach to better decoding.
    
    Args:
        log_probs: (T, B, V) log probabilities from model
        beam_width: Number of beams to keep
        
    Returns:
        List of decoded sequences (one per batch item)
    """
    T, B, V = log_probs.shape
    blank_id = 0
    results = []
    
    # Convert to probabilities for easier computation
    probs = torch.exp(log_probs.detach().cpu())  # (T, B, V)
    
    for b in range(B):
        # Beam: list of (prefix, last_char, score)
        # prefix: list of character indices
        # last_char: last non-blank character (for CTC collapse)
        # score: cumulative log probability
        beam = [([], blank_id, 0.0)]
        
        for t in range(T):
            new_beam = []
            timestep_probs = probs[t, b]  # (V,)
            
            for prefix, last_char, score in beam:
                # Try extending with each possible character
                for char_id in range(V):
                    char_prob = float(timestep_probs[char_id])
                    new_score = score + char_prob
                    
                    if char_id == blank_id:
                        # Blank: keep same prefix (CTC allows blanks)
                        new_beam.append((prefix, blank_id, new_score))
                    elif char_id == last_char:
                        # Same character: CTC collapse (merge repeated)
                        new_beam.append((prefix, char_id, new_score))
                    else:
                        # New character: extend prefix
                        new_prefix = prefix + [char_id]
                        new_beam.append((new_prefix, char_id, new_score))
            
            # Keep top beam_width candidates
            # Sort by score (higher is better)
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_width]
        
        # Return best prefix
        if beam:
            best_prefix = max(beam, key=lambda x: x[2])[0]
            results.append(best_prefix)
        else:
            results.append([])
    
    return results


def ctc_decode(log_probs: torch.Tensor, beam_width: int = 0) -> List[List[int]]:
    """
    Unified CTC decoding interface.
    
    Args:
        log_probs: (T, B, V) log probabilities from model
        beam_width: If > 0, use beam search with this width. If 0, use greedy.
        
    Returns:
        List of decoded sequences
    """
    if beam_width > 0:
        return ctc_beam_decode(log_probs, beam_width=beam_width)
    else:
        return ctc_greedy_decode(log_probs)


