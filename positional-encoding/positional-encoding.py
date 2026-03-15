import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    PE = np.zeros((seq_len,d_model))

    # for pos in range(seq_len):
    #     for i in range(0,d_model,2):
    #         denominator = base**(i/d_model)

    #         PE[pos,i] = np.sin(pos/denominator)

    #         if i+1 < d_model:
    #             PE[pos,i+1] = np.cos(pos/denominator)

    position = np.arange((seq_len))[:, np.newaxis]
    dimension = np.arange((d_model))[np.newaxis,:]

    angle_rate = 1/np.power(base,(2*(dimension//2))/d_model)

    angle_rads = position*angle_rate

    PE[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    PE[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    
    
    return PE
