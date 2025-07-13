# Adaptive Gaussian Mixture Models-Based Anomaly Detection for Under-Constrained Cable-Driven Parallel Robots

<p>
<b>Preprint available at <a href="https://arxiv.org/abs/2507.07714">https://arxiv.org/abs/2507.07714</a> </b>

<p>
<b> Abstract </b> 
<it>Cable-Driven Parallel Robots (CDPRs) are increasingly used for load manipulation tasks involving predefined toolpaths with intermediate stops. At each stop, where the platform maintains a fixed pose and the motors keep the cables under tension, the system must evaluate whether it is safe to proceed by detecting anomalies that could compromise performance (e.g., wind gusts or cable impacts). This paper investigates whether anomalies can be detected using only motor torque data, without additional sensors. It introduces an adaptive, unsupervised outlier detection algorithm based on Gaussian Mixture Models (GMMs) to identify anomalies from torque signals. The method starts with a brief calibration period—just a few seconds—during which a GMM is fit on knownanomaly-free data. Real-time torque measurements are then evaluated using Mahalanobis distance from the GMM, with statistically derived thresholds triggering anomaly flags. Model parameters are periodically updated using the latest segments identified as anomaly-free to adapt to changing conditions. Validation includes 14 long-duration test sessions simulating varied wind intensities. The proposed method achieves a 100% true positive rate and 95.7% average true negative rate, with 1-second detection latency. Comparative evaluation against power threshold and non-adaptive GMM methods indicates higher robustness to drift and environmental variation.</it>

  
## Paper materials
<ul>
  <li><b>cdpr_outlier_detector.py</b>: main script for data processing</li>
  <li><b>*.csv</b>: data files, one per experiment (limited to 25 MB due to GitHub restrictions)</li>
  <li><b>*.log</b>: log files with labeling information for each experiment</li>
  <li>Full data files available at <a href="https://dpv.uvigo.gal/index.php/s/cJWP4iWQrHq5ZrY" target="_blank">this link</a></li>
</ul>

