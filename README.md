# Adaptive A/B Testing

This project implements methods from the paper [Stronger Neyman Regret Guarantees for Adaptive
Experimental Design](https://arxiv.org/abs/2502.17427). It is built to test and compare adaptive A/B testing techniques. We compare our adaptive, strongly convex, no-variance-regret average treatment effect (ATE) estimation algorithms with the adaptive no-variance-regret algorithm from [Dai et al (2023)](https://arxiv.org/abs/2305.17187).

## Project Structure

- **abtester/**: Main library with optimizers and utility functions.
- **scripts/**: Scripts for data preprocessing, running experiments, and analysis.

## Setup

* Clone the repository.

* Navigate to the project directory and install dependencies using pip or Poetry.

## Usage

* Preprocess data:

   ```bash
   python scripts/preprocess.py
   ```

* Run experiments:

   ```bash
   python -m scripts.run_experiments
   ```
