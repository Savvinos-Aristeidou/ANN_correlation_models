# ANN correlation models
Correlation models for several IMs developed via ANN

**ANN models are available under /models folder.**

# Quick start without tensorflow models

### 1. Install requirements

```shell
pip3 install -r requirements.txt
```

### 2. Run a sample code using interpolations (example.py)

```python
from correlation_models import aso2024_correlation_int, supported_ims, \
    supported_im_pairs


# Print supported IM names
supported_ims()

# Print supported IM pairs
supported_im_pairs()

# Example 1 - SA-FIV3
period1 = 1.05
period2 = 0.5
corr = aso2024_correlation_int("SA-FIV3", period1, period2)

print(f"Correlation coefficient between SA({period1}s) "
      f"and FIV3({period2}) is {corr:.2f}!")

# Example 2 - Sa_avg2-Ds575
period1 = 0.6
corr = aso2024_correlation_int("Sa_avg2-Ds575", period1)

print(f"Correlation coefficient between Sa_avg2({period1}s) "
      f"and Ds575 is {corr:.2f}!")

```

### 3. Run a sample code using biases, weights and activation functions (example2.py)

**Recommended approach**

```python
from correlation_models import aso2024_correlation


# Example 1 - SA-FIV3
period1 = 1.05
period2 = 0.5
corr = aso2024_correlation("SA-FIV3", period1, period2)

print(f"Correlation coefficient between SA({period1}s) "
      f"and FIV3({period2}) is {corr:.3f}!")

# Example 2 - Sa_avg2-Ds575
period1 = 0.6
corr = aso2024_correlation("Sa_avg2-Ds575", period1)

print(f"Correlation coefficient between Sa_avg2({period1}s) "
      f"and Ds575 is {corr:.3f}!")
```
