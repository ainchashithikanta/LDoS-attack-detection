import numpy as np


# FSWT Parameters

lam = 1.0
eta = 0.05
upsilon = 0.5
kappa = 23.5482


# Frequency Slice Function
# Gaussian FSF
#
# p_hat(omega) = exp(-0.5 * omega^2)

def fsf(omega):

    return np.exp(-0.5 * omega ** 2)


# FSWT
#
# W(lambda, i) =
# lambda * F^-1{F{X} * P_i*}

def fswt(signal, sampling_rate):

    N = len(signal)

    # Fourier transform
    X = np.fft.fft(signal)

    # FFT frequency values
    frequencies = np.fft.fftfreq(
        N,
        d=1 / sampling_rate
    )

    # FSWT matrix
    W = np.zeros(
        (N, N),
        dtype=complex
    )

    # Create each frequency slice

    for i in range(N):

        center_frequency = frequencies[i]

        # Avoid division by zero at DC
        if center_frequency == 0:

            P = np.zeros(N)

            P[i] = 1.0

        else:

            # sigma = omega / kappa
            sigma = abs(center_frequency) / kappa

            # Frequency difference from center
            frequency_difference = (
                frequencies - center_frequency
            )

            # Normalized frequency
            omega = (
                frequency_difference / sigma
            )

            # Gaussian FSF
            P = fsf(omega)

        # FSWT equation
        #W(λ,i)=λF−1{F{X}Pi∗​}
        W[i, :] = lam * np.fft.ifft(
            X * np.conjugate(P)
        )

    return W