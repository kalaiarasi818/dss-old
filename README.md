# Carbon Footprint & Sustainability-Focused Routing DSS

A decision support prototype that estimates the CO2 emissions of a route and
flags environmentally costly ones, so logistics planners can act before dispatch.

Course project (MAT5024 Decision Support Systems, VIT Vellore), submitted as a
group of 3. I wrote the code, trained the models and built the Streamlit app.

## What it does
- Takes route inputs: distance, average speed, traffic level, road gradient,
  vehicle type and fuel efficiency
- A Random Forest Regressor predicts the carbon footprint (kg CO2)
- A Random Forest Classifier labels the route SUSTAINABLE or FLAGGED
- Shows an alert for flagged routes and a feature-importance chart that
  shows which factors drive emissions
- Includes a feedback and audit-log interface (simulated in this prototype)

## Screenshots
<img width="449" height="644" alt="route-input (1)" src="https://github.com/user-attachments/assets/77db6a40-2032-43dc-b27b-4dda8aadf6ff" />
<img width="980" height="592" alt="feature-importance" src="https://github.com/user-attachments/assets/a1f3dfe0-5de2-4ff1-8ea3-a8e0b19bd408" />
<img width="1028" height="566" alt="app-result" src="https://github.com/user-attachments/assets/a463f62f-1cad-4ed2-8cd5-ce2cc3ac9e95" />



## What drives emissions
In the trained model, average speed and fuel efficiency had the highest
feature importance, followed by distance and road gradient.

## Tech stack
Python, scikit-learn, Pandas, Streamlit, Matplotlib, Seaborn

## How to run
1. Clone the repo and open the folder
2. pip install -r requirements.txt
3. streamlit run app.py

## Notes
This is a prototype trained on a small sample dataset. Model accuracy has not
been evaluated on held-out data.
This is a prototype trained on a small sample dataset. Model accuracy has not
been evaluated on held-out data.
