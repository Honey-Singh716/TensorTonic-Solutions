import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if(max_len == None):
       L = max(len(seq) for seq in seqs)
    else:
        L = max_len
        
    N = len(seqs)

    padding = np.full((N, L), pad_value)

    for i,seq in enumerate(seqs):
        length = min(len(seq),L)
        padding[i,:length] = seq[:length]
        
    # Your code here
    return padding