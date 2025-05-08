'''
import numpy as np
import matplotlib.pyplot as plt

def simulate_bpsk(snr_db, num_bits):
    
   
    bits = np.random.randint(0, 2, num_bits)
    
    
    bpsk_signal = 2 * bits - 1  

    
    signal_power = np.mean(np.abs(bpsk_signal) ** 2)

    
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

  
    noise = np.random.normal(0, np.sqrt(noise_power), bpsk_signal.shape)

  
    received_signal = bpsk_signal + noise

   
    decoded_bits = (received_signal > 0).astype(int)

  
    errors = np.sum(bits != decoded_bits)
    ber = errors / num_bits
    return ber


num_bits = 100000  
snr_values = np.arange(0, 21, 1)  
ber_values = []


for snr in snr_values:
    ber = simulate_bpsk(snr, num_bits)
    ber_values.append(ber)
    print(f"SNR: {snr} dB, BER: {ber:.6e}")


plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, ber_values, marker='o')
plt.title('BER vs SNR for BPSK Modulation')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axis([0, 20, 1e-5, 1])
plt.show()
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_bpsk(received_signals, num_bits):
  
    bits = received_signals 

  
    bpsk_signal = 2 * bits - 1  

    
    signal_power = np.mean(np.abs(bpsk_signal) ** 2)
    
  
    snr_db = 10  
    snr_linear = 10 ** (snr_db / 10)  
    noise_power = signal_power / snr_linear

   
    noise = np.random.normal(0, np.sqrt(noise_power), bpsk_signal.shape)

   
    received_signal = bpsk_signal + noise


    decoded_bits = (received_signal > 0).astype(int)

    
    errors = np.sum(bits != decoded_bits)
    ber = errors / num_bits
    return ber


data = pd.read_csv(r'/home/wwj/chenqiliang/wubiaoqian/All-channel_matrix_p_30.csv', header=None)  
bits = data.values.flatten() 
num_bits = len(bits)  


ber = simulate_bpsk(bits, num_bits)


print(f"BER at SNR = 10 dB: {ber:.6e}")


snr_values = np.arange(0, 21, 1)  
ber_values = []

for snr in snr_values:
    ber = simulate_bpsk(bits, num_bits)  
    ber_values.append(ber)

plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, ber_values, marker='o')
plt.title('BER vs SNR for BPSK Modulation with Custom Dataset')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axis([0, 20, 1e-5, 1])
plt.show()
