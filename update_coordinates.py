import requests
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app and database
app = Flask(__name__)
app.secret_key = "asdfaksjdhfajsdhfjkashdfjkashdfjkashdfjkhajsdfkasd"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///incident_reports.db'
db = SQLAlchemy(app)

# CitizenData model
class CitizenData(db.Model):
    __tablename__ = "citizendata"
    
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ADDRESS = db.Column(db.Text, nullable=True)
    PRECINCT = db.Column(db.Text, nullable=True)
    NAME = db.Column(db.Text, nullable=False)
    GENDER = db.Column(db.String(10), nullable=True)
    BIRTHDAY = db.Column(db.String(10), nullable=True)
    BARANGAY = db.Column(db.Text, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    countrycode = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<CitizenData(ID={self.ID}, NAME='{self.NAME}', ADDRESS='{self.ADDRESS}', PRECINCT='{self.PRECINCT}', GENDER='{self.GENDER}', BIRTHDAY='{self.BIRTHDAY}', BARANGAY='{self.BARANGAY}', longitude='{self.longitude}', latitude='{self.latitude}', countrycode='{self.countrycode}')>"

# Function to geocode the address and return the latitude and longitude using OpenCage
def geocode_address(address, barangay, countrycode):
    query = f"{address}, {barangay}, {countrycode}"
    opencage_url = f"https://api.opencagedata.com/geocode/v1/json?q={query}&key=cdfa6c191e234db2972453187c4ae180&limit=1"
    
    try:
        response = requests.get(opencage_url)
        data = response.json()
        
        if data['results']:
            latitude = float(data['results'][0]['geometry']['lat'])
            longitude = float(data['results'][0]['geometry']['lng'])
            print(f"Geocoded: {query} -> Latitude: {latitude}, Longitude: {longitude}")
            return latitude, longitude
        else:
            print(f"No geocoding result for {query}")
            return None, None
    except Exception as e:
        print(f"Error in geocoding {query}: {e}")
        return None, None

# Function to update the citizen data in the database
def update_coordinates():
    citizens = CitizenData.query.filter(CitizenData.ADDRESS.isnot(None), 
                                         CitizenData.BARANGAY.isnot(None), 
                                         CitizenData.countrycode.isnot(None)).all()
    
    for citizen in citizens:
        # Skip if latitude and longitude are already available
        if citizen.latitude and citizen.longitude:
            print(f"Skipping {citizen.NAME} as coordinates are already available.")
            continue
        
        # Get geocoding results for the address, barangay, and countrycode
        latitude, longitude = geocode_address(citizen.ADDRESS, citizen.BARANGAY, citizen.countrycode)
        
        # If geocoding was successful, update the coordinates
        if latitude and longitude:
            citizen.latitude = latitude
            citizen.longitude = longitude
            db.session.commit()  # Commit the change to the database
            print(f"Updated coordinates for {citizen.NAME}")
        else:
            print(f"Could not update coordinates for {citizen.NAME}")

# Run the script
if __name__ == "__main__":
    # To test the geocoding update functionality
    with app.app_context():  # Ensure to run the function within the Flask app context
        update_coordinates()