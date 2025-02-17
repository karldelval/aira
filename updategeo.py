import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float
import os
from flask import Flask, request, jsonify, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import time
import random

# Initialize Flask app and database
app = Flask(__name__)
app.secret_key = "asdfaksjdhfajsdhfjkashdfjkashdfjkashdfjkhajsdfkasd"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///incident_reports.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder for media uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SQLAlchemy with Flask
db = SQLAlchemy(app)

# Define the CitizenData model
class CitizenData(db.Model):
    __tablename__ = "citizendata"
    
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Primary key
    ADDRESS = db.Column(db.Text, nullable=True)  # Address can be NULL
    PRECINCT = db.Column(db.Text, nullable=True)  # Precinct can be NULL
    NAME = db.Column(db.Text, nullable=False)  # Name is required
    GENDER = db.Column(db.String(10), nullable=True)  # Gender as a short string
    BIRTHDAY = db.Column(db.String(10), nullable=True)  # Birthday in text format (e.g., "YYYY-MM-DD")
    BARANGAY = db.Column(db.Text, nullable=True)  # Barangay can be NULL
    LATITUDE = db.Column(db.Float, nullable=True)  # Latitude of address
    LONGITUDE = db.Column(db.Float, nullable=True)  # Longitude of address

    def __repr__(self):
        return f"<CitizenData(ID={self.ID}, NAME='{self.NAME}', ADDRESS='{self.ADDRESS}', PRECINCT='{self.PRECINCT}', GENDER='{self.GENDER}', BIRTHDAY='{self.BIRTHDAY}', BARANGAY='{self.BARANGAY}')>"

# Initialize geolocator (using OpenStreetMap Nominatim)
geolocator = Nominatim(user_agent="citizen_geocoder")

def generate_random_coordinates_within_quezon_city():
    """
    Generates random latitude and longitude within the bounds of Quezon City.
    Approximate bounds for Quezon City:
    Latitude: 14.5° to 15.2°
    Longitude: 121.0° to 122.0°
    """
    latitude = random.uniform(14.5, 15.2)
    longitude = random.uniform(121.0, 122.0)
    return latitude, longitude

def update_lat_long_for_address(address, retries=3, timeout=10):
    """
    Returns latitude and longitude for a given address using Nominatim.
    Retries if the geocode request times out or fails.
    If the geocoding fails, returns coordinates within Quezon City bounds.
    """
    for attempt in range(retries):
        try:
            if address:
                location = geolocator.geocode(address, timeout=timeout)
                if location:
                    return location.latitude, location.longitude
                else:
                    print(f"Address not found: {address}")
                    return generate_random_coordinates_within_quezon_city()
        except GeocoderTimedOut:
            print(f"Geocoding timed out for address: {address}. Retrying... ({attempt + 1}/{retries})")
            time.sleep(2)  # Wait before retrying
        except GeocoderUnavailable as e:
            print(f"Geocoder service unavailable: {str(e)}. Retrying... ({attempt + 1}/{retries})")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

    print(f"Failed to get geolocation for address: {address} after {retries} retries. Using default coordinates.")
    return generate_random_coordinates_within_quezon_city()

def update_citizen_data():
    # Query to get the citizen data with addresses that are not NULL and no coordinates
    citizens = db.session.query(CitizenData).filter(CitizenData.ADDRESS.isnot(None), 
                                                    (CitizenData.LATITUDE == None) | (CitizenData.LONGITUDE == None)).all()

    for citizen in citizens:
        address = citizen.ADDRESS
        print(f"Processing address: {address}")

        # Get latitude and longitude for the address
        latitude, longitude = update_lat_long_for_address(address)

        # Update the latitude and longitude in the database
        citizen.LATITUDE = latitude
        citizen.LONGITUDE = longitude
        db.session.commit()  # Commit the changes to the database
        print(f"Updated: {address} with Lat: {latitude}, Long: {longitude}")

    print("Update process complete.")

# Run the update process
if __name__ == "__main__":
    with app.app_context():  # Ensure the Flask app context is available for DB operations
        update_citizen_data()