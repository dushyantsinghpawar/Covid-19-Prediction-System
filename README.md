# COVID-19 Prediction App

This project is a COVID-19 prediction app that utilizes the Prophet forecasting model from Facebook to predict the number of COVID-19 cases in India. The app provides predictions for confirmed cases, active cases, recovered cases, and deaths. 

## Project Structure

```
project_root/
│
├── README.md
├── app.py
├── coronav.jpg
├── covid_19_data[2].csv
├── forecast_confirm_model.pckl
├── forecast_active_model.pckl
├── forecast_recovered_model.pckl
├── forecast_death_model.pckl
├── index.html
├── predict.html
├── project_active.py
├── project_death.py
├── project_recover.py
├── project_total.py
└── style.css
```

- `README.md`: Readme file with project information.
- `app.py`: Main Flask application file.
- `coronav.jpg`: Background image for the web application.
- `covid_19_data[2].csv`: Dataset containing COVID-19 data.
- `forecast_confirm_model.pckl`, `forecast_active_model.pckl`, `forecast_recovered_model.pckl`, `forecast_death_model.pckl`: Pickle files containing pre-trained Prophet forecasting models.
- `index.html` and `predict.html`: HTML templates for the web application.
- `project_active.py`, `project_death.py`, `project_recover.py`, `project_total.py`: Python files to create models and pickle files for different predictions.
- `style.css`: Cascading Style Sheets (CSS) file for styling the web application.

## Setup Instructions

1. Run each `project_*.py` file separately to create pickle files for confirmed cases, active cases, recovered cases, and deaths.
2. Place the dataset file (`covid_19_data[2].csv`) in the project root directory.
3. Ensure the background image (`coronav.jpg`) is available in the root directory.
4. Add the HTML files (`index.html` and `predict.html`) and the CSS file (`style.css`) to the root directory.

## Running the App

1. Open a terminal in the project's root directory.
2. Run the Flask application:

   ```bash
   python app.py
   ```

3. Access the app through your web browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## Contributor

- **Bhagyesh Tiwari**

Feel free to customize the project structure, styles, or any other aspect based on your preferences. Ensure that you have the required dependencies installed (Flask, pandas, etc.) before running the application.
