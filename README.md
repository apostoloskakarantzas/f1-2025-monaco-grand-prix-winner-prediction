# F1 2025 Monaco GP Winner Prediction

First ML side project outside of college — applied what I learned from my vehicle price prediction project to F1. Random Forest classifier trained on Monaco race data from Kaggle (2000–2020).

## Stack

- Python
- scikit-learn (Random Forest + adjusted classification threshold)
- pandas / numpy / matplotlib / seaborn

## Data

Monaco race data from Kaggle (2000–2020), merged across results, qualifying, pit stops, lap times, drivers, and constructors tables.

## Features

`grid`, `qualifying_position`, `grid_gap_to_pole`, `driver_win_rate`, `driver_avg_finish_last3`, `driver_monaco_races`, `constructor_avg_points`, `constructor_monaco_win_rate`, `constructor_recent_points`, `pit_stop_count`, `avg_lap_time`

## Models

Random Forest classifier with `class_weight='balanced'` to handle the heavy class imbalance (1 winner per race). Threshold adjusted to 0.3 for better recall on the positive class.

## Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Main.ipynb
```

## Notes

- Train/test split is time-based: trained on races up to 2022, tested on 2023+
- Dataset only goes up to 2020 so 2021–2024 data is not included — 2025 predictions use hardcoded driver/grid estimates
- Missing values imputed with median (numeric) and mode (categorical)

## Future Improvements

- Extend dataset beyond 2020 and experiment with how many years of data actually improve vs hurt the model
- Add weather and safety car deployment as features
- Rethink whether Monaco-only training data is better than all-track data given how unique the circuit is

## License

MIT
