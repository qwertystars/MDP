#!/usr/bin/env python3
"""
Structural Health Monitoring - Data Analysis Pipeline
Processes data from ESP32 sensor array and performs comprehensive analysis

Features:
- Load and parse CSV data from SD cards
- FFT analysis with frequency shift detection
- Wavelet decomposition for damage indicators
- Machine learning anomaly detection
- Report generation with visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SHMDataAnalyzer:
    """
    Comprehensive structural health monitoring data analyzer
    """
    
    def __init__(self, baseline_file=None, sample_rate=200):
        self.sample_rate = sample_rate
        self.baseline_data = None
        self.baseline_stats = None
        
        if baseline_file:
            self.load_baseline(baseline_file)
    
    def load_data(self, csv_file):
        """
        Load data from ESP32 CSV log
        
        Expected columns:
        timestamp, s0_x, s0_y, s0_z, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z,
        ens_x, ens_y, ens_z, t0, t1, t2
        """
        print(f"Loading data from {csv_file}...")
        
        df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype != 'datetime64[ns]':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"Loaded {len(df)} samples ({len(df)/self.sample_rate:.1f} seconds)")
        print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        return df
    
    def compute_fft(self, signal_data, axis_name='X'):
        """
        Compute FFT and identify dominant frequencies
        """
        n = len(signal_data)
        
        # Apply Hanning window
        window = np.hanning(n)
        windowed = signal_data * window
        
        # Compute FFT
        fft_vals = fft(windowed)
        freqs = fftfreq(n, 1/self.sample_rate)
        
        # Take only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_magnitude = np.abs(fft_vals[pos_mask])
        
        # Find peaks (structural modes)
        peaks, properties = signal.find_peaks(
            fft_magnitude,
            height=np.max(fft_magnitude) * 0.1,  # 10% of max
            distance=int(0.5 * self.sample_rate / 100)  # At least 0.5 Hz apart
        )
        
        peak_freqs = freqs[peaks]
        peak_mags = fft_magnitude[peaks]
        
        # Sort by magnitude
        sorted_idx = np.argsort(peak_mags)[::-1]
        peak_freqs = peak_freqs[sorted_idx[:5]]  # Top 5 peaks
        peak_mags = peak_mags[sorted_idx[:5]]
        
        return {
            'freqs': freqs,
            'magnitude': fft_magnitude,
            'peak_freqs': peak_freqs,
            'peak_mags': peak_mags,
            'dominant_freq': peak_freqs[0] if len(peak_freqs) > 0 else 0
        }
    
    def compute_time_domain_features(self, signal_data):
        """
        Calculate time-domain statistical features
        """
        return {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak': np.max(np.abs(signal_data)),
            'peak_to_peak': np.ptp(signal_data),
            'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
            'kurtosis': stats.kurtosis(signal_data),
            'skewness': stats.skew(signal_data)
        }
    
    def wavelet_decomposition(self, signal_data, wavelet='db6', level=6):
        """
        Perform wavelet decomposition and calculate detail energies
        """
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Calculate energy in each detail level
        energies = []
        for i, coeff in enumerate(coeffs[1:], start=1):
            energy = np.sum(coeff**2)
            energies.append(energy)
        
        # Approximation energy
        approx_energy = np.sum(coeffs[0]**2)
        
        # Calculate frequency ranges for each level
        freq_ranges = []
        for i in range(1, level+1):
            freq_high = self.sample_rate / (2**i)
            freq_low = self.sample_rate / (2**(i+1))
            freq_ranges.append((freq_low, freq_high))
        
        return {
            'coeffs': coeffs,
            'detail_energies': energies,
            'approx_energy': approx_energy,
            'freq_ranges': freq_ranges
        }
    
    def establish_baseline(self, df):
        """
        Establish baseline statistics from healthy structure data
        Should be run on 24+ hours of normal operation data
        """
        print("\nEstablishing baseline statistics...")
        
        baseline_stats = {}
        
        for axis in ['x', 'y', 'z']:
            ens_col = f'ens_{axis}'
            data = df[ens_col].values
            
            # Time domain features
            td_features = self.compute_time_domain_features(data)
            
            # FFT features
            fft_data = self.compute_fft(data, axis.upper())
            
            # Wavelet features
            wavelet_data = self.wavelet_decomposition(data)
            
            baseline_stats[axis] = {
                'time_domain': td_features,
                'dominant_freq': fft_data['dominant_freq'],
                'peak_freqs': fft_data['peak_freqs'].tolist(),
                'detail_energies': wavelet_data['detail_energies'],
                'temperature_range': (df['t0'].min(), df['t0'].max())
            }
            
            print(f"  Axis {axis.upper()}:")
            print(f"    RMS: {td_features['rms']*1000:.2f} mg")
            print(f"    Dominant frequency: {fft_data['dominant_freq']:.2f} Hz")
            print(f"    Kurtosis: {td_features['kurtosis']:.2f}")
        
        self.baseline_stats = baseline_stats
        self.baseline_data = df
        
        return baseline_stats
    
    def save_baseline(self, filename='baseline_stats.json'):
        """Save baseline statistics to file"""
        if self.baseline_stats is None:
            raise ValueError("No baseline established. Run establish_baseline() first.")
        
        with open(filename, 'w') as f:
            json.dump(self.baseline_stats, f, indent=2)
        
        print(f"\nBaseline saved to {filename}")
    
    def load_baseline(self, filename):
        """Load baseline statistics from file"""
        with open(filename, 'r') as f:
            self.baseline_stats = json.load(f)
        
        print(f"Baseline loaded from {filename}")
    
    def detect_anomalies(self, df, thresholds=None):
        """
        Detect anomalies by comparing current data to baseline
        
        Returns DataFrame with anomaly flags and scores
        """
        if self.baseline_stats is None:
            raise ValueError("No baseline available. Load or establish baseline first.")
        
        if thresholds is None:
            thresholds = {
                'rms_factor': 1.5,      # 150% of baseline
                'freq_shift': 0.02,     # 2% shift
                'kurtosis': 5.0,        # Absolute threshold
                'crest_factor': 1.5     # 150% of baseline
            }
        
        print("\nPerforming anomaly detection...")
        
        results = {
            'timestamp': [],
            'axis': [],
            'anomaly_detected': [],
            'anomaly_type': [],
            'confidence': [],
            'details': []
        }
        
        # Analyze in windows (e.g., every 1024 samples = ~5 seconds)
        window_size = 1024
        n_windows = len(df) // window_size
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = df.iloc[start_idx:end_idx]
            
            for axis in ['x', 'y', 'z']:
                ens_col = f'ens_{axis}'
                data = window[ens_col].values
                
                # Compute current features
                td_features = self.compute_time_domain_features(data)
                fft_data = self.compute_fft(data, axis.upper())
                
                # Get baseline for this axis
                baseline = self.baseline_stats[axis]
                
                # Check for anomalies
                anomalies = []
                
                # RMS check
                rms_factor = td_features['rms'] / baseline['time_domain']['rms']
                if rms_factor > thresholds['rms_factor']:
                    anomalies.append(f"RMS {rms_factor:.2f}× baseline")
                
                # Frequency shift check
                freq_shift = (fft_data['dominant_freq'] - baseline['dominant_freq']) / baseline['dominant_freq']
                if abs(freq_shift) > thresholds['freq_shift']:
                    anomalies.append(f"Freq shift {freq_shift*100:.1f}%")
                
                # Kurtosis check (early fault indicator)
                if td_features['kurtosis'] > thresholds['kurtosis']:
                    anomalies.append(f"High kurtosis {td_features['kurtosis']:.2f}")
                
                # Crest factor check
                cf_factor = td_features['crest_factor'] / baseline['time_domain']['crest_factor']
                if cf_factor > thresholds['crest_factor']:
                    anomalies.append(f"Crest factor {cf_factor:.2f}× baseline")
                
                # Record results
                if len(anomalies) > 0:
                    results['timestamp'].append(window['timestamp'].iloc[0])
                    results['axis'].append(axis.upper())
                    results['anomaly_detected'].append(True)
                    results['anomaly_type'].append(', '.join(anomalies))
                    results['confidence'].append(min(1.0, len(anomalies) * 0.3))
                    results['details'].append({
                        'rms_factor': float(rms_factor),
                        'freq_shift_pct': float(freq_shift * 100),
                        'kurtosis': float(td_features['kurtosis']),
                        'current_freq': float(fft_data['dominant_freq'])
                    })
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"\n*** {len(results_df)} ANOMALIES DETECTED ***")
            print(results_df.head(10))
        else:
            print("\nNo anomalies detected. Structure appears healthy.")
        
        return results_df
    
    def generate_report(self, df, anomalies_df=None, output_dir='reports'):
        """
        Generate comprehensive analysis report with visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'shm_report_{timestamp}.html'
        
        print(f"\nGenerating report: {report_file}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Time series plot
        ax1 = plt.subplot(4, 2, 1)
        for axis in ['x', 'y', 'z']:
            ax1.plot(df.index / self.sample_rate, 
                    df[f'ens_{axis}'] * 1000, 
                    label=f'{axis.upper()}-axis', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (mg)')
        ax1.set_title('Time Series - Ensemble Average')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. FFT plots for each axis
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = plt.subplot(4, 2, 2 + i)
            
            data = df[f'ens_{axis}'].values
            fft_data = self.compute_fft(data, axis.upper())
            
            ax.plot(fft_data['freqs'], fft_data['magnitude'])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'FFT Spectrum - {axis.upper()}-axis')
            ax.set_xlim(0, 50)  # Focus on structural frequency range
            ax.grid(True, alpha=0.3)
            
            # Mark dominant frequency
            if len(fft_data['peak_freqs']) > 0:
                ax.axvline(fft_data['dominant_freq'], 
                          color='r', linestyle='--', alpha=0.7,
                          label=f'Dominant: {fft_data['dominant_freq']:.2f} Hz')
                ax.legend()
        
        # 3. Sensor comparison
        ax5 = plt.subplot(4, 2, 5)
        time_samples = df.index[:1000] / self.sample_rate  # First 5 seconds
        ax5.plot(time_samples, df['s0_x'][:1000] * 1000, label='Sensor 0', alpha=0.6)
        ax5.plot(time_samples, df['s1_x'][:1000] * 1000, label='Sensor 1', alpha=0.6)
        ax5.plot(time_samples, df['s2_x'][:1000] * 1000, label='Sensor 2', alpha=0.6)
        ax5.plot(time_samples, df['ens_x'][:1000] * 1000, 
                label='Ensemble', linewidth=2, color='black')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Acceleration (mg)')
        ax5.set_title('Sensor Comparison (X-axis, first 5s)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 4. Temperature plot
        ax6 = plt.subplot(4, 2, 6)
        ax6.plot(df.index / self.sample_rate, df['t0'], label='Sensor 0')
        ax6.plot(df.index / self.sample_rate, df['t1'], label='Sensor 1')
        ax6.plot(df.index / self.sample_rate, df['t2'], label='Sensor 2')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Temperature (°C)')
        ax6.set_title('Sensor Temperature')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 5. Wavelet decomposition
        ax7 = plt.subplot(4, 2, 7)
        data = df['ens_x'].values[:2048]  # Use 2048 samples
        wavelet_data = self.wavelet_decomposition(data)
        
        levels = range(1, len(wavelet_data['detail_energies']) + 1)
        energies = wavelet_data['detail_energies']
        
        ax7.bar(levels, energies)
        ax7.set_xlabel('Wavelet Detail Level')
        ax7.set_ylabel('Energy')
        ax7.set_title('Wavelet Detail Energy Distribution')
        ax7.grid(True, alpha=0.3)
        
        # Add frequency range labels
        for i, (low, high) in enumerate(wavelet_data['freq_ranges'], start=1):
            ax7.text(i, energies[i-1], f'{low:.1f}-{high:.1f}Hz', 
                    ha='center', va='bottom', fontsize=8)
        
        # 6. Statistical summary
        ax8 = plt.subplot(4, 2, 8)
        ax8.axis('off')
        
        summary_text = "=== Statistical Summary ===\n\n"
        
        for axis in ['x', 'y', 'z']:
            data = df[f'ens_{axis}'].values
            features = self.compute_time_domain_features(data)
            fft_data = self.compute_fft(data)
            
            summary_text += f"{axis.upper()}-axis:\n"
            summary_text += f"  RMS: {features['rms']*1000:.2f} mg\n"
            summary_text += f"  Peak: {features['peak']*1000:.2f} mg\n"
            summary_text += f"  Crest Factor: {features['crest_factor']:.2f}\n"
            summary_text += f"  Kurtosis: {features['kurtosis']:.2f}\n"
            summary_text += f"  Dominant Freq: {fft_data['dominant_freq']:.2f} Hz\n\n"
        
        if self.baseline_stats:
            summary_text += "=== Baseline Comparison ===\n\n"
            for axis in ['x', 'y', 'z']:
                data = df[f'ens_{axis}'].values
                features = self.compute_time_domain_features(data)
                baseline = self.baseline_stats[axis]
                
                rms_factor = features['rms'] / baseline['time_domain']['rms']
                summary_text += f"{axis.upper()}-axis RMS: {rms_factor:.2f}× baseline\n"
        
        if anomalies_df is not None and len(anomalies_df) > 0:
            summary_text += f"\n=== ANOMALIES ===\n"
            summary_text += f"Total detected: {len(anomalies_df)}\n"
            summary_text += f"Axes affected: {', '.join(anomalies_df['axis'].unique())}\n"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                fontfamily='monospace', fontsize=9,
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f'shm_plots_{timestamp}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {plot_file}")
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>SHM Analysis Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .alert {{ background-color: #ffcccc; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .info {{ background-color: #ccffcc; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Structural Health Monitoring Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Data duration:</strong> {len(df)/self.sample_rate:.1f} seconds ({len(df)} samples)</p>
            <p><strong>Sample rate:</strong> {self.sample_rate} Hz</p>
            
            <h2>Analysis Results</h2>
            <img src="{plot_file.name}" alt="Analysis Plots">
            
            <h2>Anomaly Detection</h2>
        """
        
        if anomalies_df is not None and len(anomalies_df) > 0:
            html_content += f'<div class="alert"><strong>WARNING:</strong> {len(anomalies_df)} anomalies detected!</div>'
            html_content += anomalies_df.to_html(index=False)
        else:
            html_content += '<div class="info">No anomalies detected. Structure appears healthy.</div>'
        
        html_content += """
            <h2>Recommendations</h2>
            <ul>
                <li>Continue monitoring for trend analysis</li>
                <li>Re-establish baseline if building conditions change</li>
                <li>Inspect structure if persistent anomalies detected</li>
                <li>Recalibrate sensors quarterly</li>
            </ul>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {report_file}")
        
        return report_file

def main():
    """Example usage"""
    print("=== SHM Data Analysis Pipeline ===\n")
    
    # Initialize analyzer
    analyzer = SHMDataAnalyzer(sample_rate=200)
    
    # Load data
    data_file = input("Enter CSV data file path (or press Enter for demo): ").strip()
    
    if not data_file:
        print("\nDemo mode: Creating synthetic data...")
        # Create demo data
        n_samples = 10000
        t = np.linspace(0, n_samples/200, n_samples)
        
        demo_data = {
            'timestamp': pd.date_range(start='2025-01-01', periods=n_samples, freq='5ms'),
            'ens_x': 0.005 * np.sin(2*np.pi*3.5*t) + np.random.normal(0, 0.002, n_samples),
            'ens_y': 0.004 * np.sin(2*np.pi*3.3*t) + np.random.normal(0, 0.002, n_samples),
            'ens_z': 0.006 * np.sin(2*np.pi*3.7*t) + np.random.normal(0, 0.002, n_samples),
            's0_x': 0.005 * np.sin(2*np.pi*3.5*t) + np.random.normal(0, 0.003, n_samples),
            's0_y': 0.004 * np.sin(2*np.pi*3.3*t) + np.random.normal(0, 0.003, n_samples),
            's0_z': 0.006 * np.sin(2*np.pi*3.7*t) + np.random.normal(0, 0.003, n_samples),
            's1_x': 0.005 * np.sin(2*np.pi*3.5*t) + np.random.normal(0, 0.003, n_samples),
            's1_y': 0.004 * np.sin(2*np.pi*3.3*t) + np.random.normal(0, 0.003, n_samples),
            's1_z': 0.006 * np.sin(2*np.pi*3.7*t) + np.random.normal(0, 0.003, n_samples),
            's2_x': 0.005 * np.sin(2*np.pi*3.5*t) + np.random.normal(0, 0.003, n_samples),
            's2_y': 0.004 * np.sin(2*np.pi*3.3*t) + np.random.normal(0, 0.003, n_samples),
            's2_z': 0.006 * np.sin(2*np.pi*3.7*t) + np.random.normal(0, 0.003, n_samples),
            't0': np.full(n_samples, 25.0) + np.random.normal(0, 0.5, n_samples),
            't1': np.full(n_samples, 25.5) + np.random.normal(0, 0.5, n_samples),
            't2': np.full(n_samples, 24.8) + np.random.normal(0, 0.5, n_samples)
        }
        
        df = pd.DataFrame(demo_data)
    else:
        df = analyzer.load_data(data_file)
    
    # Establish baseline
    print("\nEstablishing baseline (this may take a moment)...")
    baseline = analyzer.establish_baseline(df)
    analyzer.save_baseline('baseline_stats.json')
    
    # Detect anomalies
    anomalies = analyzer.detect_anomalies(df)
    
    # Generate report
    report_file = analyzer.generate_report(df, anomalies)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Open {report_file} in your browser to view the full report")

if __name__ == '__main__':
    main()
