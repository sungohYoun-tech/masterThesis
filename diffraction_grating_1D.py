from rcwa import Source, Layer, LayerStack, Crystal, Solver, RectangularGrating
from rcwa.shorthand import complexArray
import numpy as np


def solve_system():

    reflection_layer = Layer(er=1.0, ur=1.0)
    transmission_layer = Layer(er=9.0, ur=1.0)

    wavelength = 0.5
    deg = np.pi / 180
    k0 = 2*np.pi/wavelength
    theta = 60 * deg
    phi = 1*deg
    pTEM = 1/np.sqrt(2)*complexArray([1,1j])
    source = Source(wavelength=wavelength, theta=theta, phi=phi, pTEM=pTEM, layer=reflection_layer)

    crystal_thickness = 0.5

    N_harmonics = 11

    grating_layer = RectangularGrating(period=2, thickness=0.5, n=4, n_void=1, nx=500)
    layer_stack = LayerStack(grating_layer, incident_layer=reflection_layer, transmission_layer=transmission_layer)

    solver_1d = Solver(layer_stack, source, N_harmonics)
    results = solver_1d.solve()

    return results

if __name__ == '__main__':
    results = solve_system()

    # Get the amplitude reflection and transmission coefficients
    (rxCalculated, ryCalculated, rzCalculated) = (results['rx'], results['ry'], results['rz'])
    (txCalculated, tyCalculated, tzCalculated) = (results['tx'], results['ty'], results['tz'])

    # Get the diffraction efficiencies R and T and overall reflection and transmission coefficients R and T
    (R, T, RTot, TTot) = (results['R'], results['T'], results['RTot'], results['TTot'])
    print(RTot, TTot, RTot+TTot)

    # ------------------------------------------------------------------
    # ADDED: Amplitude, Phase, and Polarization Analysis for 0th Order
    # ------------------------------------------------------------------

    # 1. Find the index for the 0th order harmonic (Direct transmission)
    # The harmonics array is centered, so the middle index corresponds to the 0th order.
    center_index = len(txCalculated) // 2

    # 2. Extract the complex Electric Field (E-field) for the 0th order
    Ex_0 = txCalculated[center_index]
    Ey_0 = tyCalculated[center_index]

    # 3. Calculate Amplitude (Magnitude of the complex number)
    # Intensity is proportional to Amplitude squared (|E|^2)
    amp_x = np.abs(Ex_0)
    amp_y = np.abs(Ey_0)

    # 4. Calculate Phase (Angle of the complex number)
    # Returns the phase in radians between -pi and pi
    phase_x = np.angle(Ex_0)
    phase_y = np.angle(Ey_0)

    # 5. Polarization Analysis (Ellipsometry Parameters)
    # Delta (Phase Difference): difference between X and Y phase
    delta = phase_x - phase_y
    
    # Psi (Amplitude Ratio Angle): arctan of the ratio of amplitudes
    # This represents the polarization rotation/ellipticity
    psi = np.arctan(amp_x / amp_y)

    # --- Print the calculated results ---
    print("\n" + "="*40)
    print(" ANALYSIS RESULTS (0th Order Transmission)")
    print("="*40)
    
    print(f"[Amplitude]")
    print(f"  Ex Amplitude (|Ex|): {amp_x:.6f}")
    print(f"  Ey Amplitude (|Ey|): {amp_y:.6f}")
    
    print(f"\n[Phase]")
    print(f"  Ex Phase: {phase_x:.4f} rad ({np.degrees(phase_x):.2f} deg)")
    print(f"  Ey Phase: {phase_y:.4f} rad ({np.degrees(phase_y):.2f} deg)")

    print(f"\n[Polarization State]")
    print(f"  Phase Difference (Delta): {delta:.4f} rad ({np.degrees(delta):.2f} deg)")
    print(f"  Amplitude Ratio Angle (Psi): {psi:.4f} rad ({np.degrees(psi):.2f} deg)")
    print("="*40)
