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

# Example 3 - Sa_avg2-FIV3
period1 = 0.6
period2 = 1.0
corr = aso2024_correlation("Sa_avg2-FIV3", period1, period2)

print(f"Correlation coefficient between Sa_avg2({period1}s) "
      f"and FIV3({period2}s) is {corr:.3f}!")
