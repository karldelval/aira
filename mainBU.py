import os
import random
from datetime import datetime, timedelta
from flask_cors import CORS
from flask import Flask, Blueprint, request, jsonify, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_paginate import Pagination, get_page_parameter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sentence_transformers import SentenceTransformer,util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dlib
from shapely.geometry import Point, Polygon
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
from sqlalchemy import JSON

from flask import url_for
from datetime import timedelta
import torch
from sqlalchemy import func
import numpy as np
import re
from flask import session
import pandas as pd
import plotly.express as px
from flask import redirect, url_for
import openai
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from geopy.geocoders import Nominatim 
from langdetect import detect
import nltk
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import spacy
nlp = spacy.load("en_core_web_sm")

from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')

# Initialize Flask app and database
app = Flask(__name__)
app.secret_key="asdfaksjdhfajsdhfjkashdfjkashdfjkashdfjkhajsdfkasd"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///incident_reports.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024  # 24 MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder for media uploads
app.config['UPLOAD_FOLDER_POI'] = 'static/photos/POI'  # Folder for media uploads
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi','webm'}  # Allowed file types for uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)
CORS(app)
# Load OpenAI API key from environment variable for security
openai.api_key = "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"

# Initialize geolocator
geolocator = Nominatim(user_agent="incident_dashboard")

class BarangayClearance(db.Model):
    __tablename__ = "barangay_clearances"
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    citizen_id = db.Column(db.Integer, db.ForeignKey('citizendata.ID'), nullable=False)
    purpose = db.Column(db.Text, nullable=False)
    certificate_number = db.Column(db.String(50), nullable=False)
    ctc_issued_on = db.Column(db.Date, nullable=False)
    ctc_issued_at = db.Column(db.String(255), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    official_receipt_number = db.Column(db.String(50), nullable=False)
    digital_signature = db.Column(db.Text, nullable=False)
    mode_of_payment = db.Column(db.String(50), nullable=False)
    issued_on = db.Column(db.Date, nullable=False)
    issued_at = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    description = db.Column(db.Text, nullable=True)  # New Field for Description
    created_by = db.Column(db.String(50), nullable=False)
    tracking = db.Column(db.Text, nullable=True)  
    status = db.Column(db.Text, nullable=True)

    citizen = db.relationship('CitizenData', backref='barangay_clearances')

    def __repr__(self):
        return (f"<BarangayClearance(id={self.id}, citizen_id={self.citizen_id}, purpose='{self.purpose}', "
                f"certificate_number='{self.certificate_number}', issued_on='{self.issued_on}')>")



class SensorData(db.Model):
    __tablename__ = 'sensor_data'
    
    sensor_id = db.Column(db.Integer, primary_key=True)
    sensor_type = db.Column(db.String, nullable=False)
    value = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    other_data = db.Column(db.Integer, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<SensorData(sensor_id={self.sensor_id}, sensor_type='{self.sensor_type}', value={self.value}, timestamp={self.timestamp})>"


class IAQ(db.Model):
    __tablename__ = 'iAQ'

    sensor_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sensor_type = db.Column(db.String, nullable=True)
    value = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    other_data = db.Column(db.Integer, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    device_id = db.Column(db.Text, nullable=True)
    application_id = db.Column(db.Text, nullable=True)
    received_at = db.Column(db.Text, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    pir = db.Column(db.Text, nullable=True)
    light_level = db.Column(db.Float, nullable=True)
    co2 = db.Column(db.Float, nullable=True)
    tvoc = db.Column(db.Float, nullable=True)
    pressure = db.Column(db.Float, nullable=True)
    hcho = db.Column(db.Float, nullable=True)
    pm2_5 = db.Column(db.Float, nullable=True)
    pm10 = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<IAQ(sensor_id={self.sensor_id}, sensor_type='{self.sensor_type}', value={self.value}, timestamp={self.timestamp})>"


class OAQ(db.Model):
    __tablename__ = 'OAQ'

    sensor_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sensor_type = db.Column(db.String, nullable=True)
    value = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    other_data = db.Column(db.Integer, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    device_id = db.Column(db.Text, nullable=True)
    application_id = db.Column(db.Text, nullable=True)
    received_at = db.Column(db.DateTime, nullable=True)
    raw_payload = db.Column(db.Text, nullable=True)
    port = db.Column(db.Integer, nullable=True)
    bat_v = db.Column(db.Float, nullable=True)
    solarCharging = db.Column(db.Float, nullable=True)
    solarCurrent = db.Column(db.Float, nullable=True)
    battCurrent = db.Column(db.Float, nullable=True)
    solar_v = db.Column(db.Float, nullable=True)
    CO2 = db.Column(db.Float, nullable=True)
    Hum = db.Column(db.Float, nullable=True)
    Temp = db.Column(db.Float, nullable=True)
    PM01 = db.Column(db.Float, nullable=True)
    PM25 = db.Column(db.Float, nullable=True)
    PM10 = db.Column(db.Float, nullable=True)
    TVOC = db.Column(db.Float, nullable=True)
    NOX = db.Column(db.Float, nullable=True)
    breachedHumIn = db.Column(db.Text, nullable=True)
    breachedTempMin = db.Column(db.Text, nullable=True)
    breachedTempMax = db.Column(db.Text, nullable=True)
    breachedCO2Min = db.Column(db.Text, nullable=True)
    BreachedTVOCMax = db.Column(db.Text, nullable=True)
    breachedNOXMax = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<OAQ(sensor_id={self.sensor_id}, sensor_type='{self.sensor_type}', value={self.value}, timestamp={self.timestamp})>"


class Noise(db.Model):
    __tablename__ = 'Noise'

    sensor_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sensor_type = db.Column(db.String, nullable=True)
    value = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    other_data = db.Column(db.Integer, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    device_id = db.Column(db.Text, nullable=True)
    application_id = db.Column(db.Text, nullable=True)
    received_at = db.Column(db.DateTime, nullable=True)
    raw_payload = db.Column(db.Text, nullable=True)
    batt_v = db.Column(db.Float, nullable=True)
    averageNoise = db.Column(db.Float, nullable=True)
    noiseLevel = db.Column(db.Float, nullable=True)
    maxNoise = db.Column(db.Float, nullable=True)
    MinNoise = db.Column(db.Float, nullable=True)
    chargingVoltage = db.Column(db.Float, nullable=True)
    chargingCurrent = db.Column(db.Float, nullable=True)
    chargingStatus = db.Column(db.Float, nullable=True)
    transmission = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f"<Noise(sensor_id={self.sensor_id}, sensor_type='{self.sensor_type}', value={self.value}, timestamp={self.timestamp})>"

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    urgency = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    alert_method = db.Column(db.String(50), nullable=False)  # Email, SMS, Messenger
    contact_details = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=True)  # Active/Inactive status
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

    def __repr__(self):
        return f'<Alert {self.name}>'
    

class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Announcement {self.title}>'

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, nullable=False)  # Assuming users have unique IDs
    announcement_id = db.Column(db.Integer, db.ForeignKey('announcement.id'), nullable=False)

    def __repr__(self):
        return f'<Comment {self.content[:20]}>'
    
# USERS model
class USERS(db.Model):
    __tablename__ = 'USERS'
    user_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    access = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    mobile = db.Column(db.String(15), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    location_id = db.Column(db.String(50), nullable=True)
    instance_id = db.Column(db.String(50), nullable=True)
    photo = db.Column(db.String(200), nullable=True)
    role =  db.Column(db.String(20), nullable=True)

    # Database Model for Geofence
class Geofence(db.Model):
    # Your model fields
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    boundaries = db.Column(JSON)
    description = db.Column(db.Text, nullable=False) 
    area = db.Column(db.Float, nullable=True)  # Area in sqm

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'boundaries': self.boundaries,
            'description': self.description,
            'area': self.area # Just return the boundaries as a list
        }

class Analysis(db.Model):
    __tablename__ = 'analysis'

    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False, unique=True)
    analysis_text = db.Column(db.Text, nullable=False)  # Final analysis text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Analysis question_id={self.question_id}>"
    


class ResponseDB(db.Model):
    __tablename__ = 'responses'
    response_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=False)
    tag = db.Column(db.String(20), nullable=False)

class CitizenData(db.Model):
    __tablename__ = "citizendata"
    
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ADDRESS = db.Column(db.Text, nullable=True)
    PRECINCT = db.Column(db.Text, nullable=True)
    NAME = db.Column(db.Text, nullable=False)
    GENDER = db.Column(db.String(10), nullable=True)
    BIRTHDAY = db.Column(db.String(10), nullable=True)  # Format: "YYYY-MM-DD"
    BARANGAY = db.Column(db.Text, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    countrycode = db.Column(db.Text, nullable=True)
    status = db.Column(db.Text, nullable=True)

    # Relationships
    documents = db.relationship('Document', backref='citizen', lazy=True)
    kyc = db.relationship('KYC', backref='citizen', uselist=False)

    def __repr__(self):
        return (f"<CitizenData(ID={self.ID}, NAME='{self.NAME}', ADDRESS='{self.ADDRESS}', "
                f"PRECINCT='{self.PRECINCT}', GENDER='{self.GENDER}', BIRTHDAY='{self.BIRTHDAY}', "
                f"BARANGAY='{self.BARANGAY}', longitude='{self.longitude}', latitude='{self.latitude}', "
                f"countrycode='{self.countrycode}')>")

    def to_dict(self):
        return {
            "ID": self.ID,
            "ADDRESS": self.ADDRESS,
            "PRECINCT": self.PRECINCT,
            "NAME": self.NAME,
            "GENDER": self.GENDER,
            "BIRTHDAY": self.BIRTHDAY,
            "BARANGAY": self.BARANGAY,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "countrycode": self.countrycode,
            "status": self.status

        }


class Document(db.Model):
    __tablename__ = "documents"
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    citizen_id = db.Column(db.Integer, db.ForeignKey('citizendata.ID'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(500), nullable=True)  # New description field
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', citizen_id={self.citizen_id}, description='{self.description}')>"


from datetime import datetime

class KYC(db.Model):
    __tablename__ = 'kyc'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    citizen_id = db.Column(db.Integer, db.ForeignKey('citizendata.ID'), nullable=False)
    id_number = db.Column(db.String(100), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    barangay = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    
    # New fields
    occupation = db.Column(db.String(100), nullable=True)
    company = db.Column(db.String(100), nullable=True)
    nationality = db.Column(db.String(100), nullable=True)
    office_address = db.Column(db.String(255), nullable=True)
    sss = db.Column(db.String(50), nullable=True)
    tin = db.Column(db.String(50), nullable=True)
    philhealth = db.Column(db.String(50), nullable=True)
    email = db.Column(db.String(100), nullable=True)
    mobile_number = db.Column(db.String(15), nullable=True)

    # Additional fields for photo, fingerprint, facial biometric information, and different IDs
    photo = db.Column(db.String(255), nullable=True)
    fingerprint = db.Column(db.String(255), nullable=True)  # Assuming the fingerprint data is saved as a file or a hash
    facial_biometrics = db.Column(db.String(255), nullable=True)  # Store facial biometric data if needed
    id_type = db.Column(db.String(50), nullable=True)  # Type of ID (Driver's License, National ID, etc.)
    id_photo = db.Column(db.String(255), nullable=True)  # Store the ID photo file path

    def __repr__(self):
        return f"<KYC(id={self.id}, citizen_id={self.citizen_id}, is_verified={self.is_verified})>"
    
class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(10), nullable=False)

    questions = db.relationship('Question', backref='survey', lazy=True)

class Question(db.Model):
    __tablename__ = 'questions'
    
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)  # The survey question text
    question_type = db.Column(db.String(50), nullable=False, default="TEXT")  # TEXT or MULTIPLE_CHOICE
    input_method = db.Column(db.String(100), nullable=False)  # VOICE or TEXT
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp of question creation
    response_type = db.Column(db.Text, nullable=False)  # The survey question text
    # Foreign key reference to the survey table
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'), nullable=False)
    
    # Relationships
    responses = db.relationship('QResponses', backref='question', lazy=True)  # Responses relationship
    options = db.relationship('Option', backref='question', lazy=True, cascade="all, delete-orphan")  # MCQ options
    
    def __repr__(self):
        return f"<Question id={self.id} text={self.text} type={self.question_type}>"  
    
class Option(db.Model):
    __tablename__ = 'options'
    
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(255), nullable=False)  # Option text
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)  # Reference to Question

    def __repr__(self):
        return f"<Option id={self.id} text={self.text}>"


class QResponses(db.Model):
    __tablename__ = 'QResponses'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    response_text = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sentiment = db.Column(db.Text, nullable=True)
    action = db.Column(db.Text, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    location = db.Column(db.Text, nullable=True)
    name = db.Column(db.Text, nullable=True)
    address = db.Column(db.Text, nullable=True)
    colorcode = db.Column(db.Text, nullable=True)
    grouping = db.Column(db.Text, nullable=True)
    response_type = db.Column(db.Text, nullable=True)
   
    

    def __repr__(self):
        return f"<QResponses id={self.id} question_id={self.question_id} user_id={self.user_id}>"

    def to_dict(self):
        return {
            'id': self.id,
            'question_id': self.question_id,
            'user_id': self.user_id,
            'response_text': self.response_text,
            'language': self.language,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'sentiment': self.sentiment or 'neutral',  # Ensure non-null value
            'action': self.action,
            'name': self.name,
            'colorcode': self.colorcode,
            'grouping': self.grouping,
            'response_type': self.response_type,
            'address': self.address,
            'longitude': self.longitude or '0',  # Default to '0' if None
            'latitude': self.latitude or '0',  # Default to '0' if None
            'location': self.location or '',  # Default to empty string if Non       # Assuming it exists for 'similarity' mode
            
        }

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # Relief, Medical, Rescue, Barangay
    quantity = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=True)
    added_on = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Resource {self.name}>'


class IncidentAnalysis(db.Model):
    __tablename__ = 'incident_analysis'
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=False)  # Ensure foreign key is correct
    action_points = db.Column(db.String, nullable=True)
    report_text = db.Column(db.Text, nullable=False)
    tokens = db.Column(db.String, nullable=True)
    user_id = db.Column(db.Integer, nullable=False)
    

    incident = db.relationship('Incident', backref='analyses')  # Relationship with the Incident model

    
    def to_dict(self):
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "action_points": self.action_points,
            "report_text": self.report_text,
            "tokens": self.tokens,
            "user_id": self.user_id,
            
        }

class Marker(db.Model):
    __tablename__ = 'markers'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    label = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50), nullable=False)  # New column for category

    def __repr__(self):
        return f'<Marker {self.label} at ({self.latitude}, {self.longitude})>'

    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'description': self.description,
            'latitude': float(self.latitude),
            'longitude': float(self.longitude),
            'category': self.category,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class PersonOfInterest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    alias = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    last_seen_location = db.Column(db.String(200), nullable=True)
    last_seen_date = db.Column(db.DateTime, nullable=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)  # New field for photo path
    user_id = db.Column(db.Integer, nullable=False)
    
    incident = db.relationship('Incident', backref='persons_of_interest')
# Define the Incident model
# Incident Model
class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caller_name = db.Column(db.String(100))
    contact_number = db.Column(db.String(15))
    report_text = db.Column(db.Text)
    media_path = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    tokens = db.Column(db.Text, nullable=True)
    openai_analysis = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(200), nullable=True)
    language = db.Column(db.String(20), nullable=True)
    tag = db.Column(db.String(20), nullable=True)
    actionpoints = db.Column(db.Text)
    notes = db.Column(db.Text)
    type =  db.Column(db.String(20), nullable=True)
    assigned_authorities = db.Column(db.Text, nullable=True)  # New field for assigned authorities
    user_id = db.Column(db.Integer, nullable=False)
    disregard_words = db.Column(db.JSON, default=[])
    complainant =  db.Column(db.String(100), nullable=True)
    defendant =  db.Column(db.String(100), nullable=True)
     # New fields
    damage_estimate = db.Column(db.Float, nullable=True)  # For monetary value of damage
    crops_affected = db.Column(db.Text, nullable=True)    # List or description of affected crops
    recommendation = db.Column(db.Text, nullable=True)    # Recommendations for handling the incident
    field_notes = db.Column(db.Text, nullable=True)  
    

    #responses = db.relationship('Response', backref='incident_ref', lazy=True)


    def to_dict(self):
        """Convert the Incident object to a serializable dictionary."""
        return {
            'id': self.id,
            'report_text': self.report_text,
            'caller_name':self.caller_name,
            'contact_number':self.contact_number,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'tag': self.tag,
            'category': self.category,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'tokens': self.tokens,
            'openai_analysis': self.openai_analysis,
            'location': self.location,
            'actionpoints': self.actionpoints,
            'notes': self.notes,
            'type': self.type,
            'assigned_authorities': self.assigned_authorities,
            'user_id': self.user_id,
            'disregard_words': self.disregard_words,
            'complainant': self.complainant,
            'defendant':self.defendant,
            'damage_estimate': self.damage_estimate,
            'crops_affected': self.crops_affected,
            'recommendation': self.recommendation,
            'field_notes': self.field_notes,
            'media_path': self.media_path if self.media_path else None

        }
    def set_tokens(self):
        """Populate the tokens field based on report_text."""
        if self.report_text:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(self.report_text)
            self.tokens = " ".join([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]])  # Only using nouns and proper nouns as tokens

    def __repr__(self):
        return f'<Incident {self.id} - {self.category}>'
    
# Assignment Table
class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=False)
    personnel_id = db.Column(db.Integer, db.ForeignKey('USERS.user_id'), nullable=False)
    date_assigned = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='Assigned')
    notes = db.Column(db.Text)

    incident = db.relationship('Incident', backref='assignments')
    personnel = db.relationship('USERS', backref='assignments')

    def to_dict(self):
        return {
            'id': self.id,
            'incident_id': self.incident_id,
            'personnel_name': f"{self.personnel.first_name} {self.personnel.last_name}",
            'date_assigned': self.date_assigned.isoformat(),
            'status': self.status,
            'notes': self.notes
        }
    
class MessageInbox(db.Model):
    __tablename__ = 'messages'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('USERS.user_id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('USERS.user_id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='sent')  # "sent", "delivered", "read"
    message_type = db.Column(db.String(50), default='text')  # Can be 'text', 'image', 'file', etc.
    is_read = db.Column(db.Boolean, default=False)
    reply_to = db.Column(db.Integer, db.ForeignKey('messages.id'))  # Foreign key to the parent message, if replying
    attachment_url = db.Column(db.String(255))  # URL or path to an attachment

    # Relationship to the User table (sender and receiver)
    sender = db.relationship('USERS', foreign_keys=[sender_id])
    receiver = db.relationship('USERS', foreign_keys=[receiver_id])

    # Relationship to itself for replies
    parent_message = db.relationship('MessageInbox', remote_side=[id])  # For replies to a message

    def __repr__(self):
        return f"<MessageInbox {self.id} from {self.sender_id} to {self.receiver_id}>"

# Initialize database
with app.app_context():
    db.create_all()

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper functions for analysis
def fetch_incidents(filters):
    # Build dynamic query using filters
    query = Incident.query.filter(*filters)  # Filters are passed as a list of conditions
    return query.all()

import pandas as pd

def analyze_incidents(incidents):
    # Process the incidents to generate stats, handling None timestamps
    data = []
    for incident in incidents:
        if incident.timestamp is not None:
            date = incident.timestamp.date()
        else:
            date = None  # Use None or a default date, e.g., datetime(1970, 1, 1).date()
        data.append((incident.id, incident.location, incident.category, date))
    
    df = pd.DataFrame(data, columns=["id", "location", "category", "date"])

    # Analyze stats by location, category, and date
    location_stats = df.groupby('location').size().reset_index(name='incident_count')
    category_stats = df.groupby('category').size().reset_index(name='incident_count')
    date_stats = df.groupby('date').size().reset_index(name='incident_count')

    return location_stats, category_stats, date_stats





def process_predictions(predictions):
    for prediction in predictions:
        # Debugging: print forecasted incidents
        forecasted_incidents = prediction.get('incident_prediction', {}).get('forecasted_incidents_next_2_weeks', [])
        print(f"Forecasted Incidents: {forecasted_incidents}")  # Add this line to see the structure of the data

        # Remove duplicate action points
        action_points = list(set(prediction.get('incident_prediction', {}).get('action_points', [])))
        prediction['incident_prediction']['action_points'] = action_points

        # Add focus categories and incidents
        focus_data = defaultdict(int)
        
        if forecasted_incidents:
            for idx, forecast_value in enumerate(forecasted_incidents):
                # Ensure that we have both category and count data
                if isinstance(forecast_value, dict):  # Ensure that forecast is a dictionary
                    category = forecast_value.get('category')
                    count = forecast_value.get('count', 0)
                    
                    if category and isinstance(count, (int, float)):
                        focus_data[category] += count
        
        # Sort categories by count for focus
        sorted_focus = sorted(focus_data.items(), key=lambda x: x[1], reverse=True)
        prediction['focus_categories'] = sorted_focus[:3]  # Top 3 categories to focus on

        print(f"Focus Categories: {prediction['focus_categories']}")  # Add this line to debug focus categories

    return predictions

def predict_incidents(incidents):
    print(f"Incidents sample: {incidents[:5]}")
    
    # Convert incidents to DataFrame for easier manipulation
    df = pd.DataFrame([(incident.location, incident.timestamp, incident.category) 
                       for incident in incidents], columns=["location", "timestamp", "category"])

    # Preprocessing and Feature Engineering
    df['day'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year

    # 1. Calculate frequency of incidents in each location
    incident_frequency = df.groupby('location').size().reset_index(name='incident_count')

    # 2. Detect relationships between incidents using clustering (DBSCAN)
    location_category_data = df[['location', 'category']].drop_duplicates()
    location_category_data['location'] = location_category_data['location'].astype('category').cat.codes
    location_category_data['category'] = location_category_data['category'].astype('category').cat.codes
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(location_category_data)

    # DBSCAN clustering to detect patterns
    db = DBSCAN(eps=0.5, min_samples=3).fit(scaled_data)
    location_category_data['cluster'] = db.labels_

    # 3. Predict incidents in the next 2 weeks using time series forecasting (Exponential Smoothing)
    df['date'] = pd.to_datetime(df['timestamp'])
    incidents_by_date = df.groupby('date').size().reset_index(name='incident_count')
    incidents_by_date.set_index('date', inplace=True)

    # Check if there are at least 2 data points
    if len(incidents_by_date) < 2:
        return "Insufficient data to predict incidents. At least two data points are required."

    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(incidents_by_date['incident_count'], trend='add', seasonal=None)
    forecast = model.fit().forecast(steps=14)  # Predict for the next 14 days

    # 4. Detailed Interpretation of Forecast
    def interpret_forecast(forecast, threshold=2):
        """
        Interpret the forecast to identify trends.
    
        :param forecast: Pandas Series of forecasted values.
        :param threshold: The minimum change to consider a trend significant.
        :return: A description of the trend.
        """
        if len(forecast) > 0:
            # Calculate the overall change in forecast
            forecast_change = forecast.iloc[-1] - forecast.iloc[0]

            # Determine the trend based on the change and threshold
            if abs(forecast_change) < threshold:
                forecast_trend = "stable"
            elif forecast_change > 0:
                forecast_trend = "increasing"
            else:
                forecast_trend = "decreasing"

            # Determine the peak day if the trend is increasing
            peak_day = None
            if forecast_trend == "increasing":
                peak_day = forecast.idxmax()

            # Generate trend description
            if forecast_trend == "increasing":
                trend_description = (
                    f"Incident frequency is predicted to rise steadily over the next two weeks. "
                    f"The highest surge in incidents is expected around {peak_day.strftime('%Y-%m-%d') if peak_day else 'a peak date'}. "
                    f"Recommended actions: Monitor activity closely and allocate resources accordingly."
                )
            elif forecast_trend == "decreasing":
                trend_description = (
                    "Incident frequency is expected to decrease steadily over the next two weeks. "
                    "This suggests a decline in activity. Maintain regular monitoring to ensure continued stability."
                )
            else:
                trend_description = (
                    "Incident frequency is expected to remain stable over the next two weeks. "
                    "No significant changes in activity are anticipated."
                )
        else:
            trend_description = "Insufficient data to predict incident trends."

        return trend_description

    forecast_interpretation = interpret_forecast(forecast, threshold=5)

    # 5. Action points based on the prediction
    action_points = []
    for location, count in incident_frequency.values:
        if count > 10:
            action_points.append(f"Allocate additional patrols to {location}.")
        category_data = df[df['location'] == location].groupby('category').size().reset_index(name='incident_count')
        frequent_categories = category_data[category_data['incident_count'] > 5]
        for category in frequent_categories['category']:
            action_points.append(f"Organize community outreach for {category} in {location}.")

    # 6. Define start_date and end_date for the prediction period
    start_date = datetime.today().date()  # Current date as start date
    end_date = start_date + timedelta(days=14)  # 14 days ahead for end date

    # 7. Return prediction data in a structured format
    prediction_data = []
    for location, incident_count in incident_frequency.values:
        most_common_category = df[df['location'] == location]['category'].mode().iloc[0]  # Mode of the category
        prediction_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Current time as the prediction time

        prediction = {
            'location': location,
            'incident_count': incident_count,
            'most_common_category': most_common_category,
            'prediction_time': prediction_time,
            'start_date': start_date.strftime('%Y-%m-%d'),  # Add start_date to prediction
            'end_date': end_date.strftime('%Y-%m-%d'),      # Add end_date to prediction
            'incident_prediction': {
                'forecasted_incidents_next_2_weeks': forecast.tolist(),
                'forecast_interpretation': forecast_interpretation,
                'action_points': action_points
            },
            'prediction': f"High incident frequency predicted for {location}. Recommended actions: {', '.join(action_points)}"
        }
        prediction_data.append(prediction)

    # Process the predictions to add focus categories and remove duplicates
    prediction_data = process_predictions(prediction_data)

    print(f"Prediction Data: {prediction_data}")
    return prediction_data



def handle_natural_language_query(query):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes crime, missing persons, sanitation, public works, health or food security incident data. Be concise and brief only. "},
            {"role": "user", "content": f"Analyze and answer this query based on incidents data: {query}"}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Function to categorize crime text using AI
def categorize_incident(report_text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(report_text)
    filtered_text = " ".join([w for w in words if not w.lower() in stop_words])

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes various incidents such as crime, missing persons,sanitation, public works, health or food security."},
            {"role": "user", "content": f"Classify this report: {filtered_text} based on your knowledge. Do not be verbose, just use one or two words for the category."}
        ],
        max_tokens=10,
    )
    return response['choices'][0]['message']['content'].strip()

# Function to analyze report using OpenAI
def analyze_report(report_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a crime lab assistant that analyzes incident reports and provides action points based on complex algorithms and detective investigation manuals."},
            {"role": "user", "content": f"Analyze the following report and provide brief action points: {report_text}"}
        ],
        max_tokens=150,
    )
    return response['choices'][0]['message']['content'].strip()

# Function to get location from latitude and longitude
def get_location(latitude, longitude):
    try:
        location = geolocator.reverse((latitude, longitude), language='en', timeout=30)
        return location.address if location else "Unknown location"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Home route - render the report form
@app.route('/')
def home():

    announcements = Announcement.query.order_by(Announcement.timestamp.desc()).limit(5).all()
    print(announcements)
    return render_template("index.html", announcements=announcements)

@app.route('/geofence')
def geofence():
    return render_template('geofence.html')


@app.route('/inventory')
def inventory():
    resources = Resource.query.order_by(Resource.added_on.desc()).all()
    return render_template('inventory.html', resources=resources)
# Add Resource
@app.route('/add', methods=['POST'])
def add_resource():
    name = request.form['name']
    category = request.form['category']
    quantity = int(request.form['quantity'])
    description = request.form['description']
    
    new_resource = Resource(name=name, category=category, quantity=quantity, description=description)
    db.session.add(new_resource)
    db.session.commit()
    return redirect(url_for('index'))

import plotly.express as px
import plotly.io as pio
import traceback
from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta
from sqlalchemy import func
import pandas as pd
import plotly.express as px
import plotly.io as pio
import traceback
import numpy as np
from sklearn.linear_model import LinearRegression

def predictive_analysis2(incident_count):
    print("Starting predictive analysis with data shape:", incident_count.shape)
    print("Sample of input data:", incident_count.head())
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(incident_count['date']):
        incident_count['date'] = pd.to_datetime(incident_count['date'])
    
    # Create days column
    min_date = pd.to_datetime(incident_count['date'].min())  # Convert to pandas timestamp
    print("Min date:", min_date)
    
    incident_count['days'] = (incident_count['date'] - min_date).dt.days
    
    # Get unique categories
    categories = incident_count['category'].unique()
    print("Categories found:", categories)
    
    # Create future dates
    last_date = pd.to_datetime(incident_count['date'].max())  # Convert to pandas timestamp
    print("Last date:", last_date)
    
    future_dates = pd.date_range(start=last_date, periods=30, freq='D')
    future_days = (future_dates - min_date).days.values
    
    # Initialize list to store predictions
    predictions = []
    
    # For each category, fit a linear regression and make predictions
    for category in categories:
        print(f"\nProcessing category: {category}")
        cat_data = incident_count[incident_count['category'] == category]
        print(f"Data points for {category}:", len(cat_data))
        
        if len(cat_data) > 1:  # Need at least 2 points for regression
            X = cat_data['days'].values.reshape(-1, 1)
            y = cat_data['count'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            future_counts = model.predict(future_days.reshape(-1, 1))
            future_counts = np.maximum(future_counts, 0)  # Ensure non-negative counts
            
            print(f"Predictions made for {category}:", len(future_counts))
            
            # Store predictions
            for date, count in zip(future_dates, future_counts):
                predictions.append({
                    'date': date,
                    'category': category,
                    'count': round(float(count), 2)
                })
    
    print("\nTotal predictions made:", len(predictions))
    
    # Convert predictions to DataFrame
    future_df = pd.DataFrame(predictions)
    
    # Combine historical and future data
    historical_df = incident_count.copy()
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
    
    print("Final combined data shape:", combined_df.shape)
    print("Sample of final data:", combined_df.head())
    
    return combined_df


from flask import jsonify, render_template
from sqlalchemy import func
from datetime import datetime, timedelta

@app.route('/map_sensors')
def map_sensors():
    return render_template('map_sensors.html')

@app.route('/get_iaq_data')
def get_iaq_data():
    # Get the latest data for each unique location
    subquery = db.session.query(
        IAQ.longitude,
        IAQ.latitude,
        IAQ.device_id,
        func.max(IAQ.received_at).label('max_received_at')
    ).group_by(IAQ.longitude, IAQ.latitude, IAQ.device_id).subquery()

    latest_readings = db.session.query(IAQ).join(
        subquery,
        db.and_(
            IAQ.longitude == subquery.c.longitude,
            IAQ.latitude == subquery.c.latitude,
            IAQ.device_id == subquery.c.device_id,
            IAQ.received_at == subquery.c.max_received_at
        )
    ).all()

    sensor_data = []
    for reading in latest_readings:
        sensor_data.append({
            'device_id': reading.device_id,
            'latitude': float(reading.latitude) if reading.latitude else None,
            'longitude': float(reading.longitude) if reading.longitude else None,
            'temperature': reading.temperature,
            'humidity': reading.humidity,
            'co2': reading.co2,
            'tvoc': reading.tvoc,
            'pressure': reading.pressure,
            'hcho': reading.hcho,
            'pm2_5': reading.pm2_5,
            'pm10': reading.pm10,
            'light_level': reading.light_level,
            'received_at': reading.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    return jsonify(sensor_data)

@app.route('/get_iaq_history/<device_id>')
def get_iaq_history(device_id):
    # Get last 24 hours of data for the device
    past_24h = datetime.utcnow() - timedelta(hours=24)
    history = db.session.query(IAQ).filter(
        IAQ.device_id == device_id,
        IAQ.received_at >= past_24h
    ).order_by(IAQ.timestamp.asc()).all()
    
    history_data = [{
        'received_at': reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': reading.temperature,
        'humidity': reading.humidity,
        'co2': reading.co2,
        'tvoc': reading.tvoc,
        'pm2_5': reading.pm2_5,
        'pm10': reading.pm10
    } for reading in history]
    
    return jsonify(history_data)



@app.route('/get_oaq_data')
def get_oaq_data():
    subquery = db.session.query(
        OAQ.longitude,
        OAQ.latitude,
        OAQ.device_id,
        func.max(OAQ.received_at).label('max_received_at')
    ).group_by(OAQ.longitude, OAQ.latitude, OAQ.device_id).subquery()

    latest_readings = db.session.query(OAQ).join(
        subquery,
        db.and_(
            OAQ.longitude == subquery.c.longitude,
            OAQ.latitude == subquery.c.latitude,
            OAQ.device_id == subquery.c.device_id,
            OAQ.received_at == subquery.c.max_received_at
        )
    ).all()

    sensor_data = []
    for reading in latest_readings:
        sensor_data.append({
            'device_id': reading.device_id,
            'latitude': float(reading.latitude) if reading.latitude else None,
            'longitude': float(reading.longitude) if reading.longitude else None,
            'CO2': reading.CO2,
            'Hum': reading.Hum,
            'Temp': reading.Temp,
            'PM25': reading.PM25,
            'PM10': reading.PM10,
            'TVOC': reading.TVOC,
            'NOX': reading.NOX,
            'received_at': reading.received_at.strftime('%Y-%m-%d %H:%M:%S') if reading.received_at else None
        })
    return jsonify(sensor_data)

@app.route('/get_noise_data')
def get_noise_data():
    subquery = db.session.query(
        Noise.longitude,
        Noise.latitude,
        Noise.device_id,
        func.max(Noise.received_at).label('max_received_at')
    ).group_by(Noise.longitude, Noise.latitude, Noise.device_id).subquery()

    latest_readings = db.session.query(Noise).join(
        subquery,
        db.and_(
            Noise.longitude == subquery.c.longitude,
            Noise.latitude == subquery.c.latitude,
            Noise.device_id == subquery.c.device_id,
            Noise.received_at == subquery.c.max_received_at
        )
    ).all()

    sensor_data = []
    for reading in latest_readings:
        sensor_data.append({
            'device_id': reading.device_id,
            'latitude': float(reading.latitude) if reading.latitude else None,
            'longitude': float(reading.longitude) if reading.longitude else None,
            'averageNoise': reading.averageNoise,
            'noiseLevel': reading.noiseLevel,
            'maxNoise': reading.maxNoise,
            'MinNoise': reading.MinNoise,
            'received_at': reading.received_at.strftime('%Y-%m-%d %H:%M:%S') if reading.received_at else None
        })
    return jsonify(sensor_data)


@app.route('/get_geofences_citizen')
def get_geofences_citizen():
    geofences = Geofence.query.all()
    return jsonify([geofence.to_dict() for geofence in geofences])

@app.route('/get_oaq_history/<device_id>')
def get_oaq_history(device_id):
    # Get last 24 hours of data for the device
    past_24h = datetime.utcnow() - timedelta(hours=24)
    history = db.session.query(OAQ).filter(
        OAQ.device_id == device_id,
        OAQ.received_at >= past_24h
    ).order_by(OAQ.received_at.asc()).all()
    
    history_data = [{
        'received_at': reading.received_at.strftime('%Y-%m-%d %H:%M:%S') if reading.received_at else None,
        'Temp': reading.Temp,
        'Hum': reading.Hum,
        'CO2': reading.CO2,
        'TVOC': reading.TVOC,
        'PM25': reading.PM25,
        'PM10': reading.PM10,
        'NOX': reading.NOX
    } for reading in history]
    
    return jsonify(history_data)

@app.route('/get_noise_history/<device_id>')
def get_noise_history(device_id):
    # Get last 24 hours of data for the device
    past_24h = datetime.utcnow() - timedelta(hours=24)
    history = db.session.query(Noise).filter(
        Noise.device_id == device_id,
        Noise.received_at >= past_24h
    ).order_by(Noise.received_at.asc()).all()
    
    history_data = [{
        'received_at': reading.received_at.strftime('%Y-%m-%d %H:%M:%S') if reading.received_at else None,
        'averageNoise': reading.averageNoise,
        'noiseLevel': reading.noiseLevel,
        'maxNoise': reading.maxNoise,
        'MinNoise': reading.MinNoise
    } for reading in history]
    
    return jsonify(history_data)

@app.route('/commandcenter', methods=['GET'])
def commandcenter():
    try:
        # Fetch the current date
        today = datetime.utcnow().date()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_month = today.replace(day=1)

        # Fetch data for today, weekly, and monthly incidents
        incidents_today = (
            db.session.query(Incident)
            .filter(func.date(Incident.timestamp) == today)
            .order_by(Incident.category)
            .all()
        )

        incidents_weekly = (
            db.session.query(Incident)
            .filter(Incident.timestamp >= start_of_week)
            .order_by(Incident.category)
            .all()
        )

        incidents_monthly = (
            db.session.query(Incident)
            .filter(Incident.timestamp >= start_of_month)
            .order_by(Incident.category)
            .all()
        )

        # Top locations
        top_locations = db.session.query(Incident.location, func.count(Incident.id)) \
            .group_by(Incident.location).order_by(func.count(Incident.id).desc()).limit(5).all()

        # Top categories
        top_categories = db.session.query(Incident.category, func.count(Incident.id)) \
            .group_by(Incident.category).order_by(func.count(Incident.id).desc()).limit(5).all()

        # Fetch full data into a DataFrame
        incidents_query = db.session.query(Incident).statement
        engine = db.session.get_bind()
        df = pd.read_sql(incidents_query, con=engine)

        # Debug prints
        print("Before conversion:", df.dtypes)
        print(df.head())
        print("First few timestamp values:", df['timestamp'].head())

        # Handle timestamp and date conversion
        if 'timestamp' in df.columns:
            # Convert to datetime only if it's not already datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            
            # Create date column directly from timestamp
            df['date'] = df['timestamp'].dt.date
            
            # Check for any NaT values
            invalid_timestamps = df[df['timestamp'].isna()]
            if not invalid_timestamps.empty:
                print(f"Found {len(invalid_timestamps)} invalid timestamps:")
                print(invalid_timestamps)
                
            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

        # Visualization data
        figs = {}

        # Bar chart for top categories
        category_df = pd.DataFrame(top_categories, columns=['Category', 'Count'])
        category_fig = px.bar(
            category_df,
            x='Category', 
            y='Count', 
            title='Top Categories'
        )
        figs['category'] = pio.to_json(category_fig)

        # Map for locations
        if 'latitude' in df.columns and 'longitude' in df.columns:
            map_fig = px.scatter_mapbox(
                df, 
                lat='latitude', 
                lon='longitude', 
                hover_name='location',
                title='Incident Locations', 
                mapbox_style='open-street-map'
            )
            figs['map'] = pio.to_json(map_fig)

        # Predictive analysis
        if 'date' in df.columns and 'category' in df.columns:
            print("\nPreparing data for predictive analysis...")
            print("Original df shape:", df.shape)
            
            # Convert date to datetime for grouping
            df['date'] = pd.to_datetime(df['date'])
            
            # Group by date and category
            incident_count = df.groupby(['date', 'category']).size().reset_index(name='count')
            print("Grouped data shape:", incident_count.shape)
            print("Sample of grouped data:", incident_count.head())
            
            future_df = predictive_analysis2(incident_count)
            
            print("\nCreating prediction visualization...")
            prediction_fig = px.line(
                future_df, 
                x='date', 
                y='count', 
                color='category', 
                title='Incident Predictions'
            )
            figs['prediction'] = pio.to_json(prediction_fig)
            print("Visualization created successfully")

        # Debug print
        print("\nVisualization figures created:", list(figs.keys()))
        print("Figures data before template:", figs)
        
        rendered_template = render_template('commandcenter.html',
                             incidents_today=incidents_today,
                             incidents_weekly=incidents_weekly,
                             incidents_monthly=incidents_monthly,
                             top_locations=top_locations,
                             top_categories=top_categories,
                             figs=figs)
        
        print("Template rendered successfully")
        return rendered_template

    except Exception as e:
        print(f"Error in commandcenter route: {e}")
        traceback.print_exc()
        return "An error occurred", 500

def predictive_analysis(df):
    """Perform predictive analysis on incident data."""
    future_dates = [datetime.utcnow().date() + timedelta(days=i) for i in range(1, 30)]

    # Placeholder for predictions
    future_df = pd.DataFrame({'date': future_dates})
    all_categories = df['category'].unique()
    predictions = []

    for category in all_categories:
        cat_data = df[df['category'] == category]
        cat_data['days'] = (cat_data['date'] - cat_data['date'].min()).dt.days

        if len(cat_data) > 1:
            model = LinearRegression()
            model.fit(cat_data[['days']], cat_data['count'])
            future_preds = model.predict([(d - cat_data['date'].min()).days for d in future_dates])
            predictions.extend(zip(future_dates, [category] * len(future_dates), future_preds))

    future_df = pd.DataFrame(predictions, columns=['date', 'category', 'count'])
    return future_df



# Update Resource
@app.route('/update/<int:id>', methods=['POST'])
def update_resource(id):
    resource = Resource.query.get_or_404(id)
    resource.name = request.form['name']
    resource.category = request.form['category']
    resource.quantity = int(request.form['quantity'])
    resource.description = request.form['description']
    db.session.commit()
    return redirect(url_for('index'))

# Delete Resource
@app.route('/delete/<int:id>', methods=['POST'])
def delete_resource(id):
    resource = Resource.query.get_or_404(id)
    db.session.delete(resource)
    db.session.commit()
    return redirect(url_for('index'))




#Barangay Clearances
@app.route('/view_citizen_clearances/<int:citizen_id>')
def view_citizen_clearances(citizen_id):
    citizen = CitizenData.query.get_or_404(citizen_id)
    clearances = BarangayClearance.query.filter_by(citizen_id=citizen_id).all()
    return render_template('view_citizen_clearances.html', citizen=citizen, clearances=clearances)

@app.route('/edit_barangay_clearance/<int:clearance_id>', methods=['GET', 'POST'])
def edit_barangay_clearance(clearance_id):
    clearance = BarangayClearance.query.get_or_404(clearance_id)
    citizen = clearance.citizen
    
    if request.method == 'POST':
        # Update fields from form
        clearance.purpose = request.form['purpose']
        clearance.certificate_number = request.form['certificate_number']
        clearance.ctc_issued_on = request.form['ctc_issued_on']
        clearance.ctc_issued_at = request.form['ctc_issued_at']
        clearance.amount = request.form['amount']
        clearance.official_receipt_number = request.form['official_receipt_number']
        clearance.digital_signature = request.form['digital_signature']
        clearance.mode_of_payment = request.form['mode_of_payment']
        clearance.issued_on = request.form['issued_on']
        clearance.issued_at = request.form['issued_at']

        # Commit changes to database
        db.session.commit()

        flash('Barangay Clearance updated successfully!', 'success')
        return redirect(url_for('view_citizen_clearances', citizen_id=citizen.ID))
    
    return render_template('edit_barangay_clearance.html', clearance=clearance, citizen=citizen)

@app.route('/add_barangay_clearance/<int:citizen_id>', methods=['GET', 'POST'])
def add_barangay_clearance(citizen_id):
    citizen = CitizenData.query.get_or_404(citizen_id)
    
    if request.method == 'POST':
        purpose = request.form['purpose']
        certificate_number = request.form['certificate_number']
        ctc_issued_on = request.form['ctc_issued_on']
        ctc_issued_at = request.form['ctc_issued_at']
        amount = request.form['amount']
        official_receipt_number = request.form['official_receipt_number']
        digital_signature = request.form['digital_signature']
        mode_of_payment = request.form['mode_of_payment']
        issued_on = request.form['issued_on']
        issued_at = request.form['issued_at']
        created_by = 'Admin'  # Replace with the logged-in user's name if available.

        new_clearance = BarangayClearance(
            citizen_id=citizen_id,
            purpose=purpose,
            certificate_number=certificate_number,
            ctc_issued_on=ctc_issued_on,
            ctc_issued_at=ctc_issued_at,
            amount=amount,
            official_receipt_number=official_receipt_number,
            digital_signature=digital_signature,
            mode_of_payment=mode_of_payment,
            issued_on=issued_on,
            issued_at=issued_at,
            created_by=created_by
        )

        db.session.add(new_clearance)
        db.session.commit()

        flash('Barangay Clearance added successfully!', 'success')
        return redirect(url_for('view_citizen_clearances', citizen_id=citizen_id))
    
    return render_template('add_barangay_clearance.html', citizen=citizen)





@app.route('/cal_incidents', methods=['GET'])
def cal_incidents():
    # Fetch incidents and assigned personnel
    incidents = db.session.query(Incident, Assignment, USERS).join(Assignment, Incident.id == Assignment.incident_id).join(USERS, Assignment.personnel_id == USERS.user_id).all()
    
    incidents_data = []
    for incident, assignment, personnel in incidents:
        incidents_data.append({
            'id': incident.id,  # Include the incident ID
            'title': f'{incident.report_text}',  # Keep the title with the incident text
            'start': incident.timestamp,  # Use the timestamp for the event's start
            'extendedProps': {
                'assignment': {
                    'incident': incident.report_text,
                    'personnel': personnel.username
                }
            },
            'classNames': ['assigned']  # Optional: A class to style the event
        })
    
    return jsonify(incidents_data)

@app.route('/fetch_assignments', methods=['GET'])
def fetch_assignments():
    # Get the selected date from the query parameters
    date_str = request.args.get('date')
    
    # If the date is provided, filter assignments by that date
    if date_str:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d')
        assignments = db.session.query(
            Assignment, Incident, USERS
        ).join(Incident, Assignment.incident_id == Incident.id
        ).join(USERS, Assignment.personnel_id == USERS.user_id
        ).filter(Assignment.date_assigned >= selected_date,
                 Assignment.date_assigned < selected_date + timedelta(days=1)
        ).all()
    else:
        # If no date is provided, return all assignments
        assignments = db.session.query(
            Assignment, Incident, USERS
        ).join(Incident, Assignment.incident_id == Incident.id
        ).join(USERS, Assignment.personnel_id == USERS.user_id
        ).all()
    
    assignments_data = []
    
    for assignment, incident, personnel in assignments:
        assignments_data.append({
            'id': assignment.id,  # Include the Assignment ID here
            'incident': incident.report_text,
            'personnel': personnel.username,
            'timestamp': assignment.date_assigned.strftime('%Y-%m-%d %H:%M:%S')  # Format timestamp for JSON
        })
    
    return jsonify(assignments_data)

@app.route('/fetch_all_incidents', methods=['GET'])
def fetch_all_incidents():
    # Fetch all incidents without assignment details
    incidents = db.session.query(Incident).all()
    
    incidents_data = []
    for incident in incidents:
        incidents_data.append({
            'id': incident.id,  # Include the incident ID
            'title': incident.report_text,  # Use report_text for the autocomplete
        })
    
    return jsonify(incidents_data)

@app.route('/assign', methods=['POST'])
def assign_personnel():
    data = request.get_json()
    incident_id = data.get('incident_id')
    print("Received Data:", data)
    print("Incident ID:", incident_id)
    personnel_id = data.get('personnel_id')
    print("Personnel ID:", personnel_id)
    clicked_date = data.get('date')  # Get the clicked date from the request
    print("Clicked Date:", clicked_date)

    if not incident_id or not personnel_id or not clicked_date:
        return jsonify({'message': 'Missing incident, personnel ID, or clicked date'}), 400

    try:
        # Convert the clicked date to a datetime object (assuming the format is 'YYYY-MM-DD HH:MM:SS')
        clicked_date = datetime.strptime(clicked_date, '%Y-%m-%d %H:%M:%S')

        # Create a new assignment and add it to the session
        new_assignment = Assignment(
            incident_id=incident_id, 
            personnel_id=personnel_id, 
            date_assigned=clicked_date,  # Use clicked_date as the assignment date
            status='Assigned'  # Optionally, you can set the default status
        )
        db.session.add(new_assignment)
        db.session.commit()  # Commit the transaction

        return jsonify({'message': 'Personnel assigned successfully!'})
    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        print("Error:", str(e))
        return jsonify({'message': 'Error assigning personnel', 'error': str(e)}), 500


@app.route('/portal')
def portal():
    return render_template('portal.html')


# Serve the calendar UI
@app.route('/calendar_assign')
def calendar_assign():
    return render_template('calendar_assign.html')

@app.route('/remove_assignment/<int:assignment_id>', methods=['DELETE'])
def remove_assignment(assignment_id):
    try:
        # Query the assignment by ID
        assignment = Assignment.query.get(assignment_id)
        
        if not assignment:
            return jsonify({"error": "Assignment not found"}), 404

        # Delete the assignment
        db.session.delete(assignment)
        db.session.commit()

        return jsonify({"message": "Assignment removed successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/personnel', methods=['GET'])
def get_personnel():
    """
    Endpoint to fetch personnel with a specific role.
    Query Example: /personnel?role=officer
    """
    role = request.args.get('role', default=None, type=str)

    if not role:
        return jsonify({"error": "Role is required"}), 400

    # Query the USERS table for users with the specified role
    personnel = USERS.query.filter_by(role=role).all()

    # Serialize the results
    personnel_list = [
        {"user_id": user.user_id, "name": f"{user.first_name} {user.last_name}"}
        for user in personnel
    ]

    return jsonify(personnel_list)

from shapely.geometry import Polygon
from shapely.validation import explain_validity
from flask import jsonify, request

@app.route('/add_geofence', methods=['POST'])
def add_geofence():
    data = request.json
    name = data.get('name')
    description = data.get('description', '')  # Default to empty if not provided
    boundaries = data.get('boundaries')  # Expecting list of [lat, lon] pairs

    if not name or not boundaries:
        return jsonify({'error': 'Name and boundaries are required'}), 400

    try:
        # Ensure correct format for boundaries (lon, lat)
        formatted_boundaries = [(lon, lat) for lat, lon in boundaries]

        # Validate polygon
        polygon = Polygon(formatted_boundaries)
        if not polygon.is_valid:
            return jsonify({'error': f'Invalid polygon: {explain_validity(polygon)}'}), 400

        # Calculate area using Turf.js-compatible method (simplistic approximation for now)
        area = polygon.area * (111139 ** 2)  # Convert degrees to meters approximation

        # Create and save geofence
        geofence = Geofence(name=name, description=description, boundaries=boundaries, area=area)
        db.session.add(geofence)
        db.session.commit()

        return jsonify({'success': 'Geofence added successfully', 'area': area}), 201

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

import os
import openai
from flask import Response, jsonify
import math
from flask import current_app
@app.route("/analyze_question/<int:question_id>", methods=["GET"])
def analyze_question(question_id):
    # Load the question and responses from the database
    question = Question.query.get_or_404(question_id)
    responses = QResponses.query.filter_by(question_id=question.id).all()

    if not responses:
        return jsonify({"error": "No responses found for this question."}), 400

    # Combine all responses into a single text block
    responses_text = "\n".join([f"User {r.user_id or 'Anonymous'}: {r.response_text}" for r in responses])

    # Chunk the responses if they are too long for OpenAI's API
    MAX_TOKENS = 1500
    response_chunks = [responses_text[i:i + MAX_TOKENS] for i in range(0, len(responses_text), MAX_TOKENS)]
    
    # Create or retrieve the existing analysis
    existing_analysis = Analysis.query.filter_by(question_id=question.id).first()
    if not existing_analysis:
        existing_analysis = Analysis(question_id=question.id, analysis_text="")
        db.session.add(existing_analysis)
        db.session.commit()

    # Initialize the analysis text
    final_analysis = ""

    # Analyze each chunk using OpenAI
    for index, chunk in enumerate(response_chunks):
        try:
            openai.api_key = "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"
            # Prepare the prompt for OpenAI
            messages = [
                {"role": "system", "content": "You are an AI assistant that analyzes user responses."},
                {"role": "user", "content": f"Question: {question.text}\n\nAnalyze the following user responses:\n{chunk}\n\nProvide insights, themes, and any recommendations."}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500
            )

            # Append the analysis to the final text
            analysis_text = response['choices'][0]['message']['content'].strip()
            final_analysis += f"Chunk {index+1} Analysis:\n{analysis_text}\n\n"

        except Exception as e:
            return jsonify({"error": f"OpenAI analysis failed: {str(e)}"}), 500

    # Update the analysis in the database
    existing_analysis.analysis_text = final_analysis if final_analysis else "No analysis provided"
    db.session.commit()

    return jsonify({"message": "Analysis completed successfully", "analysis_text": final_analysis})
# Dummy storage for markers and geofences

# Route to render the map marker page
@app.route('/mapmarker')
def mapmarker():
    return render_template('mapmarker.html')

@app.route('/get_markers', methods=['GET'])
def get_markers():
    # Fetch all markers from the database
    markers = Marker.query.all()
    marker_list = [{
        "id": marker.id,
        "label": marker.label,
        "description": marker.description,
        "latitude": marker.latitude,
        "longitude": marker.longitude,
        "category": marker.category
    } for marker in markers]
    return jsonify({"markers": marker_list})

@app.route('/add_marker', methods=['POST'])
def add_marker():
    data = request.json
    new_marker = Marker(
        label=data['label'],
        description=data['description'],
        latitude=data['latitude'],
        longitude=data['longitude'],
        category=data['category']  # Save category data
    )
    db.session.add(new_marker)
    db.session.commit()
    return jsonify({"success": "Marker added successfully."})

@app.route('/delete_marker/<int:marker_id>', methods=['DELETE'])
def delete_marker(marker_id):
    marker = Marker.query.get(marker_id)
    if marker:
        db.session.delete(marker)
        db.session.commit()
        return jsonify({"success": "Marker deleted successfully."})
    return jsonify({"error": "Marker not found."}), 404

@app.route('/edit_marker/<int:marker_id>', methods=['PUT'])
def edit_marker(marker_id):
    marker = Marker.query.get(marker_id)
    if not marker:
        return jsonify({"error": "Marker not found."}), 404

    data = request.json
    marker.label = data.get('label', marker.label)
    marker.description = data.get('description', marker.description)
    marker.latitude = data.get('latitude', marker.latitude)
    marker.longitude = data.get('longitude', marker.longitude)
    db.session.commit()
    return jsonify({"success": "Marker updated successfully."})


@app.route("/read_analysis/<int:question_id>", methods=["GET"])
def read_analysis(question_id):
    # Get the analysis or return 404
    analysis = Analysis.query.filter_by(question_id=question_id).first()

    if not analysis:
        return jsonify({"error": "No analysis found for this question."}), 404

    # Render the HTML template and pass the question_id to it
    return render_template("analysis_page.html", question_id=question_id, analysis_text=analysis.analysis_text)


# Route: Edit a geofence
@app.route('/edit_geofence/<int:id>', methods=['POST'])
def edit_geofence(id):
    data = request.json
    name = data.get('name')
    boundaries = data.get('boundaries')  # Expecting list of lat/lng pairs

    if not name or not boundaries:
        return jsonify({'error': 'Name and boundaries are required'}), 400

    # Find the geofence to update
    geofence = Geofence.query.get(id)
    if not geofence:
        return jsonify({'error': 'Geofence not found'}), 404

    geofence.name = name
    geofence.boundaries = str(boundaries)
    db.session.commit()

    return jsonify({'success': 'Geofence updated successfully'}), 200



# Route: Delete a geofence
@app.route('/delete_geofence/<int:id>', methods=['DELETE'])
def delete_geofence(id):
    # Find the geofence to delete
    geofence = Geofence.query.get(id)
    if not geofence:
        return jsonify({'error': 'Geofence not found'}), 404

    db.session.delete(geofence)
    db.session.commit()

    return jsonify({'success': 'Geofence deleted successfully'}), 200


@app.route('/incidents_in_geofence/<int:geofence_id>', methods=['GET'])
def incidents_in_geofence(geofence_id):
    # Fetch the geofence by ID
    geofence = Geofence.query.get_or_404(geofence_id)
    geofence_name = request.args.get('name')
    
    # Create a Polygon object using the geofence boundaries
    boundaries = geofence.boundaries
    # Convert the boundary coordinates to tuples of (longitude, latitude)
    boundaries = [(lon, lat) for lat, lon in boundaries]  # Ensure the order is (lon, lat)
    polygon = Polygon(boundaries)

    # Query all incidents that have latitude and longitude
    incidents = Incident.query.filter(Incident.latitude.isnot(None), Incident.longitude.isnot(None)).all()
    incidents_in_geofence = []

    # Check if each incident's point is within the geofence
    for incident in incidents:
        point = Point(incident.longitude, incident.latitude)  # Shapely uses (longitude, latitude)

        if polygon.contains(point):
            incidents_in_geofence.append({
                'id': incident.id,
                'category': incident.category,
                'location': incident.location,
                'timestamp': incident.timestamp,
                'report_text': incident.report_text
            })

    # Return the incidents as JSON, or render in HTML if needed
    return render_template('incidents_in_geofence.html', 
                           geofence_id=geofence_id, 
                           geofence_name=geofence_name,
                           incidents=incidents_in_geofence)


# Helper function to get citizen by ID
def get_citizen_by_id(citizen_id):
    return CitizenData.query.get(citizen_id)

# Route for Upload Document
@app.route('/upload_document/<int:citizen_id>', methods=['GET', 'POST'])
def upload_document(citizen_id):
    citizen = get_citizen_by_id(citizen_id)
    if not citizen:
        return "Citizen not found", 404

    if request.method == 'POST':
        # Process uploaded document
        file = request.files.get('document')
        description = request.form.get('description')  # Get the description from the form

        if file and file.filename:
            filename = secure_filename(file.filename)

            # Create a directory for the citizen if it doesn't exist
            citizen_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(citizen_id))
            os.makedirs(citizen_folder, exist_ok=True)

            # Save file within the citizen's folder
            file_path = os.path.join(citizen_folder, filename)
            file.save(file_path)

            # Save the document info in the database
            document = Document(
                citizen_id=citizen_id,
                filename=filename,
                file_path=file_path,
                description=description  # Save the description
            )
            db.session.add(document)
            db.session.commit()

            flash(f"Document '{filename}' uploaded successfully!", "success")
            return redirect(url_for('view_details', citizen_id=citizen_id))
        else:
            flash("No file selected or invalid file!", "danger")

    return render_template('upload_document.html', citizen=citizen)

@app.route('/update_kyc/<int:citizen_id>', methods=['GET', 'POST'])
def update_kyc(citizen_id):
    citizen = get_citizen_by_id(citizen_id)
    if not citizen:
        return "Citizen not found", 404

    kyc = KYC.query.filter_by(citizen_id=citizen_id).first()

    if request.method == 'POST':
        if not kyc:
            kyc = KYC(citizen_id=citizen_id)
            db.session.add(kyc)
        
        # Handle photo upload
        photo = request.files.get('photo')
        if photo and photo.filename:
            photo_filename = secure_filename(photo.filename)
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'kyc_photos', photo_filename)
            photo.save(photo_path)
            kyc.photo = photo_path

        # Handle fingerprint upload
        fingerprint = request.files.get('fingerprint')
        if fingerprint and fingerprint.filename:
            fingerprint_filename = secure_filename(fingerprint.filename)
            fingerprint_path = os.path.join(app.config['UPLOAD_FOLDER'], 'kyc_fingerprints', fingerprint_filename)
            fingerprint.save(fingerprint_path)
            kyc.fingerprint = fingerprint_path

        # Handle facial biometrics
        facial_biometrics = request.files.get('facial_biometrics')
        if facial_biometrics and facial_biometrics.filename:
            facial_biometrics_filename = secure_filename(facial_biometrics.filename)
            facial_biometrics_path = os.path.join(app.config['UPLOAD_FOLDER'], 'kyc_facial_biometrics', facial_biometrics_filename)
            facial_biometrics.save(facial_biometrics_path)
            kyc.facial_biometrics = facial_biometrics_path

        # Handle ID photos
        id_photo = request.files.get('id_photo')
        if id_photo and id_photo.filename:
            id_photo_filename = secure_filename(id_photo.filename)
            id_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'kyc_ids', id_photo_filename)
            id_photo.save(id_photo_path)
            kyc.id_photo = id_photo_path

        # Update other KYC details
        kyc.id_number = request.form.get('id_number')
        kyc.address = request.form.get('address')
        kyc.barangay = request.form.get('barangay')
        kyc.country = request.form.get('country')
        kyc.occupation = request.form.get('occupation')
        kyc.company = request.form.get('company')
        kyc.nationality = request.form.get('nationality')
        kyc.office_address = request.form.get('office_address')
        kyc.sss = request.form.get('sss')
        kyc.tin = request.form.get('tin')
        kyc.philhealth = request.form.get('philhealth')
        kyc.email = request.form.get('email')
        kyc.mobile_number = request.form.get('mobile_number')

        # Handle verification status
        kyc.is_verified = request.form.get('is_verified') == 'on'

        db.session.commit()

        flash("KYC details updated successfully!", "success")
        return redirect(url_for('view_details', citizen_id=citizen_id))

    return render_template('update_kyc.html', citizen=citizen, kyc=kyc)


# Route for View Details
@app.route('/view_details/<int:citizen_id>')
def view_details(citizen_id):
    citizen = get_citizen_by_id(citizen_id)
    if not citizen:
        return "Citizen not found", 404

    # Get documents associated with the citizen
    documents = Document.query.filter_by(citizen_id=citizen_id).all()

    # Debugging: Print document details to check if 'documents' contains correct data
    print(f"Documents for Citizen {citizen_id}:")
    for doc in documents:
        print(f"Document filename: {doc.filename}")

    # Get KYC details (assuming a relationship exists or fetching logic is here)
    kyc = KYC.query.filter_by(citizen_id=citizen_id).first()

    return render_template('view_details.html', citizen=citizen, documents=documents, kyc=kyc)


@app.route('/process_chat', methods=['POST'])
def process_chat():
    """
    Process user queries conversationally using GPT-4 with optimized incident data handling.
    """
    try:
        # Step 1: Extract the user query
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"response": "Please provide a valid query."}), 400

        # Step 2: Initialize or update conversation history in session
        if "conversation_history" not in session:
            session["conversation_history"] = []
        
        session["conversation_history"].append({"role": "user", "content": user_message})
        
        # Step 3: Parse the query for keywords
        keywords = parse_query_keywords(user_message)

        # Step 4: Filter incident data based on keywords and grab additional data
        filtered_incidents = filter_incidents(keywords)

        if not filtered_incidents:
            session["conversation_history"].append({
                "role": "assistant",
                "content": "No relevant incidents found based on your query."
            })
            return jsonify({"response": "No relevant incidents found based on your query."})

        # Add incident links and more details
        for incident in filtered_incidents:
            incident['link'] = f'/incident/{incident["id"]}'  # Add URL link
            # Include additional data fields if available
            incident['summary'] = incident.get("description", "No description available")
            incident['timestamp'] = incident.get("timestamp", "No timestamp provided")

        # Step 5: Prepare the GPT-4 system prompt
        system_prompt = f"""
        You are AIRA, a friendly but intelligent Incident and Data Analyst. 
        Keep your answers brief, casual, and up to 150 words. 
        Only provide incident text summaries, locations, timestamps, and links. 
        You can also give specific suggestions based on Philippines.
        If more incidents exist, direct the user to the '/map' link.

        Here are the relevant incidents:
        {json.dumps(filtered_incidents, indent=2)}

        Conversation history (for context):
        {json.dumps(session['conversation_history'], indent=2)}

        Answer the user's query:
        User's question: {user_message}
        """

        # Step 6: Add system prompt and user input to conversation history
        session["conversation_history"].append({"role": "system", "content": system_prompt})

        # Step 7: Send to GPT-4
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=session["conversation_history"],
            max_tokens=500,
            temperature=0.7
        )

        # Step 8: Parse GPT-4 response
        gpt_response = openai_response['choices'][0]['message']['content']
        
        # Append assistant response to conversation history
        session["conversation_history"].append({"role": "assistant", "content": gpt_response})

        return jsonify({"response": gpt_response})

    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({"response": "An error occurred. Please try again."}), 500


def parse_query_keywords(user_message):
    """
    Parse the user query to extract relevant keywords.
    """
    keywords = []
    query_words = user_message.lower().split()
    
    # Keyword mapping
    keyword_mapping = ["fire", "theft", "yesterday", "last week", "downtown", "pending"]

    # Match keywords
    for word in query_words:
        if word in keyword_mapping:
            keywords.append(word)
    
    return keywords


def filter_incidents(keywords):
    """
    Filter the Incident table based on extracted keywords.
    """
    query = Incident.query

    # Example filtering logic based on keywords
    if "fire" in keywords:
        query = query.filter(Incident.category.ilike("%fire%"))
    if "theft" in keywords:
        query = query.filter(Incident.category.ilike("%theft%"))
    if "yesterday" in keywords:
        yesterday = datetime.utcnow() - timedelta(days=1)
        query = query.filter(Incident.timestamp >= yesterday)
    if "last week" in keywords:
        last_week = datetime.utcnow() - timedelta(days=7)
        query = query.filter(Incident.timestamp >= last_week)
    if "downtown" in keywords:
        query = query.filter(Incident.location.ilike("%downtown%"))

    # Fetch incidents and serialize them
    filtered_data = [
        {
            "id": incident.id,
            "category": incident.category,
            "location": incident.location,
            "report_text": incident.report_text if incident.report_text else "No report available.",
            "timestamp": incident.timestamp.isoformat() if incident.timestamp else "No timestamp",
            "link": f"/incident/{incident.id}"
        }
        for incident in query.limit(10).all()
    ]
    return filtered_data
# Route to fetch incidents
@app.route('/api/recent_incidents', methods=['GET'])
def get_recent_incidents():
    incidents = Incident.query.all()
    incidents_data = [
        {
            "category": incident.category,
            "location": incident.location,
            "report_text": incident.report_text,
            "id": incident.id
        }
        for incident in incidents
    ]
    return jsonify(incidents_data)

@app.route('/check_point', methods=['POST'])
def check_point():
    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if latitude is None or longitude is None:
        return jsonify({'error': 'Latitude and longitude are required'}), 400

    point = Point(longitude, latitude)  # Shapely uses (longitude, latitude)

    geofences = Geofence.query.all()
    for geofence in geofences:
        # Assuming geofence.boundaries is already a list of tuples or lists of coordinates
        boundaries = geofence.boundaries

        # Convert the boundary coordinates to tuples of (longitude, latitude)
        boundaries = [(lon, lat) for lat, lon in boundaries]  # Ensure the order is (lon, lat)

        polygon = Polygon(boundaries)  # Create the polygon from boundaries

        if polygon.contains(point):
            return jsonify({'message': f'Point is inside geofence: {geofence.name}'})

    return jsonify({'message': 'Point does not fall into any geofence'})

@app.route('/get_geofences', methods=['GET'])
def get_geofences():
    geofences = [geofence.to_dict() for geofence in Geofence.query.all()]
    print(geofences)  # Debugging: Check the geofence data structure in the console
    return jsonify({'geofences': geofences})

@app.route('/get_incidents', methods=['GET'])
def get_incidents():
    incidents = Incident.query.all()  # Get all incidents from the database
    incidents_data = [
        {
            'id': incident.id,
            'category': incident.category,
            'location': incident.location,
            'timestamp': incident.timestamp,
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'report_text': incident.report_text
        }
        for incident in incidents
    ]
    return jsonify({'incidents': incidents_data})


@app.route('/view_messages')
def view_messages():
    user_id = session.get('user_id')
    print(user_id)
    if not user_id:
        return redirect(url_for('login'))

    user_id = int(user_id)

    # Fetch messages for the logged-in user
    messages = MessageInbox.query.filter(
        (MessageInbox.sender_id == user_id) | (MessageInbox.receiver_id == user_id)
    ).order_by(MessageInbox.timestamp.desc()).all()

    return render_template('view_messages.html', messages=messages)


@app.route('/fetch_users', methods=['GET'])
def fetch_users():
    query = request.args.get('query', '')
    if query:
        users = USERS.query.filter(
            db.or_(
                USERS.first_name.ilike(f"%{query}%"),
                USERS.last_name.ilike(f"%{query}%"),
                USERS.email.ilike(f"%{query}%")
            )
        ).all()
    else:
        users = USERS.query.all()

    user_list = [
        {"id": user.user_id, "name": f"{user.first_name} {user.last_name}", "email": user.email}
        for user in users
    ]
    
    return jsonify({"users": user_list})

@app.route('/add_announcement', methods=['GET', 'POST'])
def add_announcement():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        user_id = session.get('user_id')  # Ensure 'user_id' is fetched safely from the session
        
        # Create and save the new announcement
        new_announcement = Announcement(
            title=title, 
            content=content,
            user_id=user_id
        )
        db.session.add(new_announcement)
        db.session.commit()
        
        return redirect(url_for('add_announcement'))
    
    # Fetch all announcements to display in the table
    all_announcements = Announcement.query.order_by(Announcement.timestamp.desc()).all()
    return render_template('add_announcement.html', announcements=all_announcements)

import csv
from io import StringIO
from flask import make_response

@app.route('/grouped_survey_responses', methods=['GET', 'POST'])
def grouped_survey_responses():
    # Fetch all surveys for selection
    surveys = db.session.query(Survey).all()

    if request.method == 'POST':
        survey_id = request.form['survey_id']

        # Get distinct submission names for the given survey
        submissions = (
            db.session.query(QResponses.name)
            .join(Question, QResponses.question_id == Question.id)
            .filter(Question.survey_id == survey_id)
            .distinct()
            .all()
        )

        grouped_responses = {}
        for submission_name in submissions:
            responses = db.session.query(QResponses).filter(QResponses.name == submission_name[0]).all()
            # For each response, replace user_id with the user's full name
            for response in responses:
                if response.user_id:
                    user = db.session.query(USERS).filter_by(user_id=response.user_id).first()
                    response.user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
                else:
                    response.user_name = "Anonymous"
            grouped_responses[submission_name[0]] = responses

        # Handle CSV download
        if request.form.get('action') == 'download_csv':
            survey = db.session.query(Survey).filter_by(id=survey_id).first()
            return download_csv(survey.title, grouped_responses)

        # Fetch survey details
        survey = db.session.query(Survey).filter_by(id=survey_id).first()
        return render_template('grouped_survey_responses.html', 
                               surveys=surveys,  # Pass surveys for the dropdown
                               survey=survey,  # Pass the selected survey
                               grouped_responses=grouped_responses)  # Pass grouped responses

    # If the request is GET, render the page with available surveys
    return render_template('grouped_survey_responses.html', 
                           surveys=surveys,  # Pass surveys for the dropdown
                           grouped_responses={})  # No responses to show yet

def download_csv(survey_title, grouped_responses):
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Submission ID", "Question", "Response", "User Name", "Timestamp"])
    
    # Write data rows
    for submission_id, responses in grouped_responses.items():
        for response in responses:
            writer.writerow([
                submission_id,
                response.question.text,
                response.response_text,
                response.user_name,
                response.timestamp
            ])
    
    # Create HTTP response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={survey_title}_responses.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/survey_responses', methods=['GET', 'POST'])
def survey_responses():
    # Fetch all surveys for dropdown selection
    surveys = Survey.query.all()

    selected_survey = None
    questions = []
    responses = {}

    if request.method == 'POST':
        # Get the selected survey ID
        survey_id = request.form.get('survey_id')
        if survey_id:
            selected_survey = Survey.query.filter_by(id=survey_id).first()
            if selected_survey:
                # Fetch questions for the selected survey
                questions = Question.query.filter_by(survey_id=selected_survey.id).all()

                # Organize responses per question
                for question in questions:
                    question_responses = QResponses.query.filter_by(question_id=question.id).all()
                    responses[question.id] = question_responses

    return render_template('survey_responses.html', surveys=surveys, selected_survey=selected_survey, questions=questions, responses=responses)

@app.route('/edit_announcement/<int:id>', methods=['GET', 'POST'])
def edit_announcement(id):
    announcement = Announcement.query.get_or_404(id)
    
    if request.method == 'POST':
        announcement.title = request.form['title']
        announcement.content = request.form['content']
        announcement.gps_lat_min = request.form.get('gps_lat_min')
        announcement.gps_lat_max = request.form.get('gps_lat_max')
        announcement.gps_long_min = request.form.get('gps_long_min')
        announcement.gps_long_max = request.form.get('gps_long_max')
        announcement.viewers = request.form['viewers']
        
        db.session.commit()
        return redirect(url_for('index'))
    
    return render_template('edit_announcement.html', announcement=announcement)

@app.route('/delete_announcement/<int:id>', methods=['GET'])
def delete_announcement(id):
    announcement = Announcement.query.get_or_404(id)
    db.session.delete(announcement)
    db.session.commit()
    return redirect(url_for('announcements'))

@app.route('/announcements')
def announcement():
    announcements = Announcement.query.all()
    return render_template('announcements.html', announcements=announcements)

@app.route('/add_comment/<int:announcement_id>', methods=['POST'])
def add_comment(announcement_id):
    comment_content = request.form['comment']
    new_comment = Comment(content=comment_content, user_id=1, announcement_id=announcement_id)  # Replace 1 with the logged-in user's ID
    db.session.add(new_comment)
    db.session.commit()
    return redirect(url_for('index'))

from difflib import SequenceMatcher
import json


# Function to validate latitude and longitude
def is_valid_latitude(latitude):
    return -90 <= latitude <= 90

def is_valid_longitude(longitude):
    return -180 <= longitude <= 180

def prepare_data(data, mode, color_map):
    sanitized_data = []

    for item in data:
        filter_value = item.get('filter_value', 'unknown')
        color_code = color_map[mode].get(filter_value, "gray")
        
        # Initialize sentiment to a default value
        sentiment = 'neutral'

        if mode == "sentiment":
            sentiment = item.get('sentiment', 'neutral').lower()
            if 'positive' in sentiment:
                filter_value = 'positive'
            elif 'negative' in sentiment:
                filter_value = 'negative'
            else:
                filter_value = 'neutral'
            color_code = color_map['sentiment'].get(filter_value, "gray")
        elif mode == "similarity":
            group = item.get('group', 'unknown')
            filter_value = group  # Use the group name as the filter value
            color_code = color_map['similarity'].get(group, "gray")  # Get color based on group

        try:
            latitude = float(item.get('latitude', 0))
            longitude = float(item.get('longitude', 0))

            # Check if latitude and longitude are valid
            if not is_valid_latitude(latitude) or not is_valid_longitude(longitude):
                print(f"Error: Invalid latitude or longitude for item {item}")
                continue  # Skip this item if coordinates are invalid

            sanitized_data.append({
                'latitude': latitude,
                'longitude': longitude,
                'response_text': item.get('response_text', ''),
                'location': item.get('location', ''),
                'filter_value': filter_value,
                'color_code': color_code,  # Ensure color_code is passed for similarity
                'sentiment': sentiment  # This will always be defined now
            })
        
        except ValueError as e:
            print(f"Error: Invalid data for item {item}: {e}")
            continue  # Skip item if there's an error in conversion

    return sanitized_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import openai

from googletrans import Translator
# Load the Sentence Transformer model and SpaCy NLP model once
# Initialize SentenceTransformer and OpenAI API
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # More context-aware model
nlp = spacy.load("en_core_web_sm")

class IncidentService:
    def __init__(self, search_terms, stop_words=None, threshold=0.2):
        self.search_terms = search_terms
        self.stop_words = stop_words or []
        self.threshold = threshold
        self.translator = Translator()

    def _clean_search_terms(self):
        """Remove custom stop words from the search terms."""
        return [
            term.lower().strip() for term in self.search_terms
            if term.lower().strip() not in self.stop_words
        ]

    def _translate_query(self):
        """Translate query to English and back-translate for better matching."""
        translations = set()
        for term in self.search_terms:
            try:
                translated = self.translator.translate(term, src='auto', dest='en').text
                original_lang = self.translator.detect(term).lang
                back_translated = self.translator.translate(translated, src='en', dest=original_lang).text
                translations.update([term, translated, back_translated])
            except Exception:
                translations.add(term)  # Fallback to original term
        return list(translations)

    def _calculate_similarity(self, query_terms, texts):
        """Calculate similarity scores using embeddings."""
        query_embedding = sentence_model.encode(' '.join(query_terms), convert_to_tensor=True)
        text_embeddings = sentence_model.encode(texts, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(query_embedding, text_embeddings)[0].cpu().numpy()

        return similarity_scores

    def search_incidents(self):
        """Search incidents and return only highly similar matches."""
        incidents = Incident.query.all()
        texts = [incident.report_text or "" for incident in incidents]

        # Clean and process search terms
        self.search_terms = self._clean_search_terms()
        processed_terms = self._translate_query()

        # Calculate similarity and filter results
        similarity_scores = self._calculate_similarity(processed_terms, texts)

        # Filter results based on relaxed threshold
        filtered_results = [
            (incidents[i], score)
            for i, score in enumerate(similarity_scores) if score >= self.threshold
        ]

        # Sort results by similarity score in descending order
        filtered_results.sort(key=lambda x: x[1], reverse=True)

        # Return only the top 5 most relevant matches
        return [result[0] for result in filtered_results[:5]]


@app.route('/NLsearch', methods=['GET', 'POST'])
def nlsearch():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('nlsearch.html', error="Please enter a search query.")

        # Step 1: Translate query to English using GPT-4
        try:
            translated_query = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Translate the following query to English."},
                    {"role": "user", "content": query}
                ]
            )['choices'][0]['message']['content'].strip()
        except Exception as e:
            return render_template('nlsearch.html', error=f"Translation failed: {e}")

        # Step 2: Extract keywords and entities using NLP
        try:
            doc = nlp(translated_query)
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            entities = [ent.text for ent in doc.ents]
            search_terms = list(set(keywords + entities))
        except Exception as e:
            return render_template('nlsearch.html', error=f"NLP processing failed: {e}")

        # Step 3: Add original query for multilingual support
        search_terms.append(query)

        # Step 4: Define stop words and perform the search
        stop_words = [
            "incident", "report", "find", "data", "the", "a", "an", "i want to", "sa", "na", 
            "is", "barangay", "want", "i", "to", "ko", "gusto", "maghanap", "ka", "ng", 
            "malapit", "near", "may", "mga", "kung", "mayroong"
        ]

        # Step 5: Search incidents with relevant context
        try:
            service = IncidentService(search_terms, stop_words=stop_words, threshold=0.3)  # Adjusted threshold
            incident_results = service.search_incidents()
        except Exception as e:
            return render_template('nlsearch.html', error=f"Search failed: {e}")

        # Render results
        return render_template(
            'nlsearch.html',
            query=query,
            translated_query=translated_query,
            incidents=incident_results
        )

    return render_template('nlsearch.html')




# Route for the heatmap page
@app.route('/graph', methods=['GET', 'POST'])
def graph():
    question_id = request.form.get('question_id', 1)
    mode = request.form.get('mode', 'sentiment')

    questions = Question.query.all()
    responses = QResponses.query.filter_by(question_id=question_id).all()

    # Define color map for sentiment and similarity modes
    color_map = {
        "sentiment": {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow"
        },
        "similarity": {
            "group1": "blue", "group2": "green", "group3": "purple",
            "group4": "orange", "group5": "pink", "group6": "green",
            "group7": "yellow", "group8": "red", "group9": "brown", "group10": "grey"
        }
    }

    # Handle the 'similarity' mode
    if mode == 'similarity':
        grouped_responses = group_responses_by_similarity(responses)
        responses_data = grouped_responses
    else:
        # Ensure we are working with proper dictionary objects
        responses_data = [response.to_dict() if hasattr(response, 'to_dict') else response for response in responses]

    sanitized_data = prepare_data(responses_data, mode, color_map)

    return render_template('graph.html',
                           heatmap_data=sanitized_data,
                           questions=questions,
                           mode=mode,
                           question_id=question_id,
                           filtered_responses=sanitized_data,
                           color_map=color_map
                           )



# Route for the heatmap page
@app.route('/scoringmap', methods=['GET', 'POST'])
def scoringmap():
    question_id = request.form.get('question_id', 1)
    mode = request.form.get('mode', 'sentiment')

    # Fetch questions where response_type is "scoring"
    questions = Question.query.filter_by(response_type="scoring").all()

    # Fetch and filter responses for the given question_id
    responses = QResponses.query.filter(
        QResponses.question_id == question_id,
        QResponses.colorcode.isnot(None),
        QResponses.grouping.isnot(None)
    ).all()

    # Group and prepare data
    grouped_data = {}
    for response in responses:
        key = (response.grouping, response.colorcode)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(response.to_dict())

    # Flatten grouped data into list for the frontend
    flattened_data = []
    for (grouping, colorcode), group_responses in grouped_data.items():
        for resp in group_responses:
            resp['filter_value'] = grouping  # Used in the frontend for markers
            resp['colorcode'] = colorcode
            flattened_data.append(resp)

    # Define color map for grouping markers
    color_map = {
        "grouping": {
            "group1": "blue", "group2": "green", "group3": "purple",
            "group4": "orange", "group5": "pink", "group6": "green",
            "group7": "yellow", "group8": "red", "group9": "brown", "group10": "grey"
        }
    }

    return render_template('scoringmap.html',
                           heatmap_data=flattened_data,
                           questions=questions,
                           mode=mode,
                           question_id=question_id,
                           filtered_responses=flattened_data,
                           color_map=color_map
                           )



# Route for the heatmap page
@app.route('/heatmap', methods=['GET', 'POST'])
def heatmap():
    question_id = request.form.get('question_id', 1)
    mode = request.form.get('mode', 'sentiment')

    questions = Question.query.all()
    responses = QResponses.query.filter_by(question_id=question_id).all()

    # Define color map for sentiment and similarity modes
    color_map = {
        "sentiment": {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow"
        },
        "similarity": {
            "group1": "blue", "group2": "green", "group3": "purple",
            "group4": "orange", "group5": "pink", "group6": "green",
            "group7": "yellow", "group8": "red", "group9": "brown", "group10": "grey"
        }
    }

    # Handle the 'similarity' mode
    if mode == 'similarity':
        grouped_responses = group_responses_by_similarity(responses)
        responses_data = grouped_responses
    else:
        # Ensure we are working with proper dictionary objects
        responses_data = [response.to_dict() if hasattr(response, 'to_dict') else response for response in responses]

    sanitized_data = prepare_data(responses_data, mode, color_map)

    return render_template('heatmap.html',
                           heatmap_data=sanitized_data,
                           questions=questions,
                           mode=mode,
                           question_id=question_id,
                           filtered_responses=sanitized_data,
                           color_map=color_map
                           )




from collections import defaultdict
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the translator
translator = Translator()

# Function to translate text to English if needed
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='tl', dest='en')  # 'tl' is Tagalog, 'en' is English
        return translated.text.lower()
    except Exception as e:
        print(f"Translation failed for '{text}': {e}")
        return text.lower()  # Fallback to original text

# Function to group responses based on similarity (using text similarity)
def group_responses_by_similarity(responses):
    grouped = defaultdict(list)
    color_palette = [
        "blue", "green", "purple", "orange", "pink", 
        "yellow", "red", "brown", "grey", "cyan"
    ]
    
    # Step 1: Collect and clean response texts
    response_texts = []
    cleaned_responses = []  # Store cleaned responses mapped to valid text

    for response in responses:
        raw_text = response.response_text.strip().lower()
        if raw_text:  # Ensure text is not empty
            try:
                translated_text = translate_to_english(raw_text)
                if translated_text.strip():  # Only add if not empty
                    response_texts.append(translated_text)
                    cleaned_responses.append(response)
            except Exception as e:
                print(f"Error translating '{raw_text}': {e}")
    
    if not response_texts:  # Safety check if no valid texts exist
        print("No valid texts to process for grouping.")
        return []

    # Step 2: Vectorize the texts for similarity calculation
    try:
        vectorizer = TfidfVectorizer(stop_words='english')  # Exclude common words
        tfidf_matrix = vectorizer.fit_transform(response_texts)
    except ValueError as e:
        print(f"TFIDF Vectorization failed: {e}")
        return []

    # Step 3: Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 4: Group responses based on similarity threshold (more lenient)
    visited = [False] * len(response_texts)
    groups = []
    
    for i in range(len(response_texts)):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(len(response_texts)):
            if i != j and not visited[j] and similarity_matrix[i][j] > 0.4:  # Lower similarity threshold
                group.append(j)
                visited[j] = True
        groups.append(group)

    # Step 5: Assign colors and organize grouped responses
    grouped_responses = []
    for idx, group_indices in enumerate(groups):
        group_color = color_palette[idx % len(color_palette)]
        for i in group_indices:
            response = cleaned_responses[i]
            grouped_responses.append({
                'latitude': response.latitude,
                'longitude': response.longitude,
                'response_text': response.response_text,
                'location': response.location,
                'group': f'group{idx+1}',
                'color_code': group_color
            })

    return grouped_responses

# Route for changing password
@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'user_id' not in session:
        flash('You must be logged in to access this page.', 'danger')
        return redirect('/login')
    
    user_id = session['user_id']
    user = USERS.query.filter_by(user_id=user_id).first()

    if not user:
        flash('User not found.', 'danger')
        return redirect('/login')

    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        # Validate current password
        if not check_password_hash(user.password, current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect('/change_password')

        # Validate new password match
        if new_password != confirm_password:
            flash('New password and confirmation do not match.', 'danger')
            return redirect('/change_password')

        # Update password
        user.password = generate_password_hash(new_password)
        db.session.commit()
        flash('Password updated successfully.', 'success')
        return redirect('/login')  # Redirect to user's profile or another page

    return render_template('change_password.html', user=user)


# Route to fetch all barangays
@app.route('/get_barangays', methods=['GET'])
def get_barangays():
    barangays = db.session.query(CitizenData.BARANGAY).distinct().all()
    barangay_list = [barangay[0] for barangay in barangays if barangay[0] is not None]  # Exclude NULL values
    return jsonify(barangay_list)


@app.route('/update_coordinates', methods=['POST'])
def update_coordinates():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        
        # Check if all required fields are present
        if not data or 'id' not in data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'success': False, 'error': 'Invalid request. Missing id, latitude, or longitude.'}), 400
        
        # Extract data
        citizen_id = data['id']
        latitude = data['latitude']
        longitude = data['longitude']

        # Validate latitude and longitude
        if not isinstance(latitude, (float, int)) or not isinstance(longitude, (float, int)):
            return jsonify({'success': False, 'error': 'Latitude and longitude must be numbers.'}), 400
        
        # Fetch the citizen record from the database
        citizen = CitizenData.query.get(citizen_id)
        if not citizen:
            return jsonify({'success': False, 'error': f'Citizen with ID {citizen_id} not found.'}), 404
        
        # Update the citizen's coordinates
        citizen.latitude = latitude
        citizen.longitude = longitude
        
        # Commit the changes to the database
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Coordinates for citizen ID {citizen_id} updated successfully.',
            'updated_data': {
                'id': citizen.ID,
                'name': citizen.NAME,
                'latitude': citizen.latitude,
                'longitude': citizen.longitude,
            }
        }), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': 'Database error occurred.', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': 'An unexpected error occurred.', 'details': str(e)}), 500
    
@app.route('/citizen_dashboard', methods=['GET'])
def citizen_dashboard():
    # Pagination parameters
    page = request.args.get('page', 1, type=int)  # Default to page 1
    per_page = 100  # Show 100 citizens per page

    # Search parameters from query string
    search_params = {
        "barangay": request.args.get('barangay'),
        "address": request.args.get('address'),
        "gender": request.args.get('gender'),
        "precinct": request.args.get('precinct'),
        "location": request.args.get('location'),
        "birthday": request.args.get('birthday'),
        "name": request.args.get('name')
    }

    # Construct the filter query
    filters = []
    if search_params['barangay']:
        filters.append(CitizenData.BARANGAY.ilike(f"%{search_params['barangay']}%"))
    if search_params['address']:
        filters.append(CitizenData.ADDRESS.ilike(f"%{search_params['address']}%"))
    if search_params['gender']:
        filters.append(CitizenData.GENDER == search_params['gender'])
    if search_params['precinct']:
        filters.append(CitizenData.precinct.ilike(f"%{search_params['precinct']}%"))
    if search_params['location']:
        filters.append(or_(
            CitizenData.latitude.ilike(f"%{search_params['location']}%"),
            CitizenData.longitude.ilike(f"%{search_params['location']}%")
        ))
    if search_params['birthday']:
        filters.append(CitizenData.BIRTHDAY == search_params['birthday'])
    if search_params['name']:
        filters.append(CitizenData.NAME.ilike(f"%{search_params['name']}%"))

    # Query database with filters and pagination
    query = CitizenData.query
    if filters:
        query = query.filter(and_(*filters))

    citizens = query.order_by(CitizenData.ID).paginate(page=page, per_page=per_page)

    return render_template('citizen_dashboard.html', citizens=citizens, search_params=search_params)

@app.route('/search_users', methods=['GET'])
def search_users():
    query = request.args.get('query', '').lower()
    users = USERS.query.filter(
        (USERS.name.ilike(f"%{query}%")) | (USERS.email.ilike(f"%{query}%"))
    ).limit(10).all()

    # Return the search result as a JSON response
    return jsonify({'users': [{'id': user.id, 'name': user.name, 'email': user.email} for user in users]})

@app.route('/add_citizen', methods=['GET', 'POST'])
def add_citizen():
    if request.method == 'POST':
        data = request.form
        new_citizen = CitizenData(
            NAME=data['NAME'],
            ADDRESS=data['ADDRESS'],
            BARANGAY=data['BARANGAY'],
            PRECINCT=data.get('PRECINCT'),
            GENDER=data.get('GENDER'),
            BIRTHDAY=data.get('BIRTHDAY'),
            longitude=data.get('longitude'),
            latitude=data.get('latitude'),
            countrycode=data.get('countrycode')
        )
        db.session.add(new_citizen)
        db.session.commit()
        return redirect(url_for('citizen_dashboard'))
    return render_template('add_citizen.html')

@app.route('/edit_citizen/<int:id>', methods=['GET', 'POST'])
def edit_citizen(id):
    # Fetch the citizen by ID or return a 404 error if not found
    citizen = CitizenData.query.get_or_404(id)
    
    if request.method == 'POST':
        # Fetch form data from the request
        data = request.form
        
        # Update citizen fields with data from the form
        try:
            citizen.NAME = data.get('name', '').strip()  # Ensure name is not None
            citizen.ADDRESS = data.get('address', '').strip()
            citizen.BARANGAY = data.get('barangay', '').strip()
            citizen.PRECINCT = data.get('precinct', '').strip()  # Optional field
            citizen.GENDER = data.get('gender', '').strip()
            citizen.BIRTHDAY = data.get('birthday', '').strip()
            citizen.longitude = data.get('longitude', None)  # Optional field
            citizen.latitude = data.get('latitude', None)  # Optional field
            citizen.countrycode = data.get('countrycode', '').strip()  # Optional field
            
            # Commit changes to the database
            db.session.commit()
            
            # Redirect to the citizen dashboard after successful update
            return redirect(url_for('citizen_dashboard'))
        except Exception as e:
            # Handle any errors during the update process
            db.session.rollback()
            flash(f"An error occurred while updating the citizen: {e}", "error")
            return redirect(url_for('edit_citizen', id=id))
    
    # Render the edit citizen form
    return render_template('edit_citizen.html', citizen=citizen)
from sqlalchemy import and_

@app.route('/search_citizens', methods=['GET'])
def search_citizens():
    name = request.args.get('name', '')
    address = request.args.get('address', '')
    results = CitizenData.query.filter(
        CitizenData.NAME.like(f'%{name}%'),
        CitizenData.ADDRESS.like(f'%{address}%')
    ).all()
    return jsonify([{
        'ID': c.ID,
        'NAME': c.NAME,
        'ADDRESS': c.ADDRESS,
        'BARANGAY': c.BARANGAY,
        'latitude': c.latitude,
        'longitude': c.longitude
    } for c in results])

@app.route('/get_citizens', methods=['GET'])
def get_citizens():
    try:
        # Extract query parameters
        barangay = request.args.get('barangay', '')
        name = request.args.get('name', '')
        address = request.args.get('address', '')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 30))

        print(f"Received request with parameters: Barangay={barangay}, Name={name}, Address={address}, Page={page}, Limit={limit}")
        
        # Build query
        query = CitizenData.query
        if barangay:
            query = query.filter(CitizenData.BARANGAY == barangay)
        if name:
            query = query.filter(CitizenData.NAME.like(f'%{name}%'))
        if address:
            query = query.filter(CitizenData.ADDRESS.like(f'%{address}%'))

        # Correct pagination
        citizens = query.paginate(page=page, per_page=limit, error_out=False)
        
        print(f"Fetched {len(citizens.items)} citizens from the database.")

        # Prepare response
        response_data = {
            'records': [{
                'ID': citizen.ID,
                'name': citizen.NAME,
                'address': citizen.ADDRESS,
                'barangay': citizen.BARANGAY,
                'latitude': citizen.latitude,   # Ensure 'latitude' is lowercase
                'longitude': citizen.longitude, # Ensure 'longitude' is lowercase
            } for citizen in citizens.items],
            'totalRecords': citizens.total
        }

        return jsonify(response_data)
    except Exception as e:
        # Log the exception with detailed info
        print(f"Error fetching citizen data: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print the full traceback to the console
        return jsonify({"error": "Failed to fetch citizen data"}), 500

# Route to render the citizen map page
@app.route('/citizen_map')
def map_citizens():
    return render_template('citizen_map.html')

@app.route('/notify_users', methods=['POST'])
def notify_users():
    data = request.get_json()
    user_ids = data.get('user_ids', [])
    incident_id = data.get('incident_id')

    if not user_ids:
        return jsonify({"message": "No users selected!"}), 400

    # Get the users to notify
    notified_users = USERS.query.filter(USERS.user_id.in_(user_ids)).all()
    
    # Retrieve the incident report text from the database (assuming an Incident model)
    incident = Incident.query.filter_by(id=incident_id).first()

    if not incident:
        return jsonify({"message": "Incident not found!"}), 404

    # Loop through each user to create a notification message
    for user in notified_users:
        # Create a new message in the Message table for the notification
        new_message = MessageInbox(
            sender_id=int(session['user_id']),  # The ID of the admin or system sender
            receiver_id=user.user_id,
            message=f"You have been notified about incident {incident_id}. Location: {incident.longitude}, {incident.latitude}. Here are the details:\n\n{incident.report_text}",
            status='sent',
            message_type='text',
            is_read=False,  # Initially, the message is unread
            attachment_url=None,  # You can add attachment URL if needed
        )
        
        # Add the message to the session and commit it
        db.session.add(new_message)
    
    # Commit all messages to the database
    db.session.commit()

    # Notify the users (mock logic)
    for user in notified_users:
        print(f"Notification sent to {user.first_name} {user.last_name} - Incident ID: {incident_id}")

    return jsonify({"message": f"Notifications sent to {len(notified_users)} users!"})


from uuid import uuid4

@app.route("/answer_survey", methods=["GET", "POST"])
def answer_survey():

    if 'user_id' not in session:  # Check if user_id is in the session
        return redirect(url_for('login')) 
    
    if request.method == "POST":

        survey_id = request.form['survey_id']
        latitude = request.form.get('latitude')  # Get latitude from the form
        longitude = request.form.get('longitude')  # Get longitude from the form
     
        name = str(uuid4())
        address = request.form.get('address')
        user_id = int(session['user_id'])  # Get the user_id from the form
        # Collect answers and process them
        answers = {}
        for question in Question.query.filter_by(survey_id=survey_id).all():
            answer_text = request.form.get(f"answer_{question.id}")
            answers[question.id] = answer_text

        # Save responses to the database
        for question_id, answer_text in answers.items():
            response = QResponses(
                question_id=question_id,
                response_text=answer_text,
                name=name,
                user_id=user_id,
                address=address,
                timestamp=datetime.utcnow(),
                latitude=latitude,  # Save latitude with the response
                longitude=longitude  # Save longitude with the response
            )
            db.session.add(response)

        db.session.commit()
        return redirect(url_for('survey_submitted'))

    surveys = Survey.query.filter_by(status="Active").all()
    return render_template('answer_survey.html', surveys=surveys)

@app.route("/survey_submitted")
def survey_submitted():
    return render_template('survey_submitted.html')




@app.route('/create_survey', methods=['GET', 'POST'])
def create_survey():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        status = request.form['status']
        new_survey = Survey(title=title, description=description, status=status)
        db.session.add(new_survey)
        db.session.commit()
        return redirect('/create_survey')
    # Fetch all surveys to display in the template
    surveys = Survey.query.all()
    return render_template('create_survey.html', surveys=surveys)

@app.route('/edit_survey/<int:survey_id>', methods=['GET', 'POST'])
def edit_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    if request.method == 'POST':
        survey.title = request.form['title']
        survey.description = request.form['description']
        survey.status = request.form['status']
        db.session.commit()
        return redirect('/create_survey')
    return render_template('edit_survey.html', survey=survey)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Route to view IAQ data
@app.route('/view_iaq', methods=['GET'])
def view_iaq():
    iaq_data = IAQ.query.all()
    return render_template('view_iaq.html', iaq_data=iaq_data)

# Route to view OAQ data
@app.route('/view_oaq', methods=['GET'])
def view_oaq():
    oaq_data = OAQ.query.all()
    return render_template('view_oaq.html', oaq_data=oaq_data)

# Route to view Noise data
@app.route('/view_noise', methods=['GET'])
def view_noise():
    noise_data = Noise.query.all()
    return render_template('view_noise.html', noise_data=noise_data)

@app.route('/view_responses')
def view_responses():
    surveys = Survey.query.all()
    with db.session.no_autoflush:  # Disable autoflush during processing
        for survey in surveys:
            survey.questions = Question.query.filter_by(survey_id=survey.id).all()
            for question in survey.questions:
                question.responses = QResponses.query.filter_by(question_id=question.id).all()
                for response in question.responses:
                    if not response.sentiment or not response.action:
                        sentiment_result = compute_sentiment(response.response_text)

                        # Validate sentiment_result before proceeding
                        if "label" in sentiment_result and "score" in sentiment_result:
                            response.sentiment = f"{sentiment_result['label']} ({sentiment_result['score']})"
                        else:
                            response.sentiment = "Error: Sentiment analysis failed."

                        # Handle action result
                        action_result = "Analysis Pending."
                        if isinstance(action_result, tuple):
                            response.action = action_result[0]  # Take the first element (e.g., '200 OK')
                        else:
                            response.action = str(action_result)  # Or any default logic to convert to string
                        
                        print(f"Formatted Sentiment: {response.sentiment}")
                        print(f"Action: {response.action}")
                        db.session.add(response)  # Stage changes for commit
        db.session.commit()  # Commit changes to the database
    return render_template('view_responses.html', surveys=surveys)

from googletrans import Translator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
translator = Translator()

from googletrans import Translator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
translator = Translator()

def compute_sentiment(text):
    """Compute sentiment using VADER and TextBlob after translating the text to English."""
    try:
        # Translate the text to English if it's not in English already
        translated_text = translator.translate(text, src='auto', dest='en').text

        # Compute VADER sentiment score
        vader_result = vader_analyzer.polarity_scores(translated_text)
        vader_score = vader_result["compound"]

        # Compute TextBlob sentiment score
        blob = TextBlob(translated_text)
        blob_score = blob.sentiment.polarity

        # Combine both scores with a weighted average (e.g., 70% VADER, 30% TextBlob)
        combined_score = 0.7 * vader_score + 0.3 * blob_score

        # Classify sentiment based on combined score
        aggregated_label = classify_aggregated_sentiment(combined_score)

        return {
            "label": aggregated_label,
            "score": round(combined_score, 2),  # Use combined score for the result
            "vader_score": round(vader_score, 2),  # Optional: include individual scores for transparency
            "blob_score": round(blob_score, 2)   # Optional: include individual scores for transparency
        }

    except Exception as e:
        # Handle errors, e.g., translation errors, connection issues
        return {"error": str(e)}

def classify_aggregated_sentiment(score):
    """Classify sentiment based on aggregated score."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def generate_action_points2(text):
    """Generate action points using OpenAI."""
    try:
        api_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a government assistant generating action points for user feedback."},
                {"role": "user", "content": f"Based on this feedback: '{text}', provide recommended action points in a concise format."}
            ]
        )
        action_points = api_response["choices"][0]["message"]["content"]
        print (action_points)
        return action_points.strip()  # Return the content as a string
    except Exception as e:
        print(f"Error generating action points: {e}")
        return "No action points available."  # Return a string even on error

def classify_aggregated_sentiment(aggregated_score):
    """Classify the sentiment based on the aggregated sentiment score."""
    positive_threshold = 0.6
    negative_threshold = 0.4

    if aggregated_score >= positive_threshold:
        return "Positive"
    elif aggregated_score <= negative_threshold:
        return "Negative"
    else:
        return "Neutral"

# Load spaCy's English model
@app.template_filter('datetimeformat')
def datetimeformat(value):
    if isinstance(value, datetime):  # Now you can directly use datetime
        return value.strftime('%Y-%m-%d %H:%M:%S')  # Adjust the format as needed
    return value


# Load BERT model and tokenizer
# Initialize BERT tokenizer and model for embedding generation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
# Assuming get_bert_embeddings is replaced by a sentence transformer model or similar
def get_bert_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Better for semantic search
    return model.encode(text)

@app.route('/manage_questions', methods=['GET'])
def manage_questions():
    # Fetch all surveys with questions
    surveys = Survey.query.all()  # Ensure surveys are fetched
    if not surveys:
        print("No surveys found!")  # Debugging message
    for survey in surveys:
        survey.questions = Question.query.filter_by(survey_id=survey.id).all()
    return render_template('manage_questions.html', surveys=surveys)

    return render_template('manage_questions.html', surveys=surveys)

@app.route('/add_question', methods=['POST'])
def add_question():
    survey_id = request.form['survey_id']
    question_text = request.form['text']
    question_type = request.form['question_type']
    input_method = request.form['input_method']
    
    # Create the new question
    new_question = Question(
        survey_id=survey_id,
        text=question_text,
        question_type=question_type,
        input_method=input_method
    )
    db.session.add(new_question)
    db.session.commit()
    
    # If it's a multiple-choice question, handle the options
    if question_type == 'MULTIPLE_CHOICE':
        options = request.form.getlist('options[]')  # Fetch all options from form
        for option_text in options:
            if option_text.strip():  # Avoid empty options
                new_option = Option(
                    question_id=new_question.id,
                    text=option_text
                )
                db.session.add(new_option)
        db.session.commit()

    return redirect('/manage_questions')

@app.route('/edit_question/<int:question_id>', methods=['GET', 'POST'])
def edit_question(question_id):
    question = Question.query.get_or_404(question_id)
    if request.method == 'POST':
        question.text = request.form.get('text')
        db.session.commit()
        return redirect(url_for('manage_questions'))
    return render_template('edit_question.html', question=question)

@app.route('/delete_question/<int:question_id>', methods=['GET'])
def delete_question(question_id):
    # Fetch the question
    question = Question.query.get_or_404(question_id)
    
    # Delete associated responses first
    QResponses.query.filter_by(question_id=question.id).delete()
    
    # Delete the question
    db.session.delete(question)
    db.session.commit()
    return redirect(url_for('manage_questions'))

@app.route('/toggle_survey_status/<int:survey_id>', methods=['POST'])
def toggle_survey_status(survey_id):
    survey = Survey.query.get(survey_id)
    if survey:
        survey.status = 'Inactive' if survey.status == 'Active' else 'Active'
        db.session.commit()
    return redirect('/create_survey')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        access = "active"

        # Hash the password before saving
        hashed_password = generate_password_hash(password)

        new_user = USERS(
            first_name=first_name,
            last_name=last_name,
            email=email,
            mobile=mobile,
            username=username,
            password=hashed_password,
            role=role
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('User added successfully!', 'success')
            return redirect(url_for('manage_users'))  # Redirect to a page listing all users
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')

    return render_template('manage_users.html')  # render the page with the form

@app.route('/update_role/<int:user_id>', methods=['POST'])
def update_role(user_id):
    user = db.session.get(USERS, user_id)
    if user:
        user.role = request.form['role']
        db.session.commit()
        flash('User role updated successfully!', 'success')
    else:
        flash('User not found.', 'danger')
    return redirect(url_for('manage_users'))

import sqlite3  # Add this import at the top of your file
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@app.route('/generate_action_points/<int:incident_id>', methods=['GET'])
def generate_action_points(incident_id):
    # Fetch the related incident
    incident = Incident.query.get(incident_id)
    if not incident:
        return jsonify({"success": False, "error": "Incident not found."}), 404

    if 'user_id' not in session:
        return jsonify({"success": False, "error": "User not authenticated."}), 401

    report_text = incident.report_text

    try:
        # Call OpenAI GPT-4 API to generate action points
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant generating actionable steps for incident reports. Use Philippine setting. Be brief and direct. Add authorities that need to be contacted."},
                {"role": "user", "content": f"Generate action points for this incident report: {report_text}"}
            ]
        )
        
        try:
            action_points = response['choices'][0]['message']['content']
            tokens_used = response.get('usage', {}).get('total_tokens', 0)
        except (KeyError, IndexError) as e:
            return jsonify({"success": False, "error": f"Invalid response from OpenAI: {str(e)}"}), 500

        # Ensure values are properly extracted and not tuples
        action_points = str(action_points)  # Convert action points to string
        tokens_used = int(tokens_used)     # Ensure tokens is an integer
        report_text = str(report_text)     # Ensure report text is a string
        user_id = int(session['user_id'])  # Ensure user_id is an integer

        # Create a new IncidentAnalysis entry
        analysis = IncidentAnalysis(
            incident_id=incident_id,           # Should be an integer, not a tuple
            report_text=report_text,           # Should be a string
            action_points=action_points,       # Should be a string
            tokens=tokens_used,                # Should be an integer
            user_id=user_id                    # Should be an integer
        )
        db.session.add(analysis)
        db.session.commit()

        return redirect(request.referrer)

    except openai.error.OpenAIError as e:
        return jsonify({"success": False, "error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/send_message/<int:receiver_id>', methods=['POST'])
def send_message(receiver_id):
    try:
        request_data = request.get_json()
        print(f"Received data: {request_data}")

        sender_id = session['user_id']
        message_text = request_data.get('message')

        if not sender_id or not message_text:
            print("Error: Missing sender_id or message")
            return jsonify({"error": "Sender ID and message are required"}), 400

        # Create new message object
        new_message = MessageInbox(sender_id=sender_id, receiver_id=receiver_id, message=message_text)
        print(f"Created message object: {new_message}")

        # Add message object to the session and commit
        db.session.add(new_message)

        # Retry mechanism for database insert
        retries = 3
        for _ in range(retries):
            try:
                db.session.commit()
                print(f"Message saved with id: {new_message.id}")
                
                # Return a success response with message details
                return jsonify({
                    "success": True,
                    "message": "Message sent successfully",
                    "data": {
                        "sender_id": new_message.sender_id,
                        "receiver_id": new_message.receiver_id,
                        "message": new_message.message,
                        "id": new_message.id
                    }
                }), 200
            except sqlite3.OperationalError as e:
                print(f"Error: {e}")
                time.sleep(1)
        
        # If commit fails, return an error
        return jsonify({"error": "Database is locked. Please try again later."}), 500

    except Exception as e:
        print(f"Error sending message: {e}")
        return jsonify({"error": "Failed to send message", "details": str(e)}), 500
       

from flask import request, render_template
from flask_paginate import Pagination, get_page_parameter
from sqlalchemy import or_

@app.route('/users', methods=['GET', 'POST'])
def manage_users():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 100
    search_query = request.args.get('search', '').strip()

    # Build the query dynamically
    users_query = USERS.query
    if search_query:
        users_query = users_query.filter(
            or_(
                USERS.first_name.ilike(f'%{search_query}%'),
                USERS.last_name.ilike(f'%{search_query}%'),
                USERS.email.ilike(f'%{search_query}%')
            )
        )

    # Paginate results
    paginated_users = users_query.paginate(page=page, per_page=per_page, error_out=False)

    # Extract items and build pagination object
    users = paginated_users.items
    pagination = Pagination(
        page=page,
        total=paginated_users.total,
        search=bool(search_query),
        per_page=per_page,
        css_framework='bootstrap4'
    )

    return render_template(
        'manage_users.html',
        users=users,
        pagination=pagination,
        search_query=search_query
    )

# Route to activate/deactivate user
@app.route('/toggle_user/<int:user_id>', methods=['POST'])
def toggle_user(user_id):
    user = USERS.query.get(user_id)
    if not user:
        flash('User not found!', 'danger')
        return redirect(url_for('manage_users'))

    # Toggle access field
    user.access = 'active' if user.access != 'active' else 'inactive'
    db.session.commit()
    flash(f"User {user.first_name} {user.last_name}'s status changed to {user.access}.", 'success')
    return redirect(url_for('manage_users'))

@app.route('/view_announcements')
def view_announcements():
    # Fetch all announcements from the database
    announcements = Announcement.query.all()
    return render_template('view_announcements.html', announcements=announcements)

@app.route('/get_messages/<int:user_id>', methods=['GET'])
def get_messages(user_id):
    try:
        messages = MessageInbox.query.filter(
            (MessageInbox.sender_id == user_id) | (MessageInbox.receiver_id == user_id)
        ).all()

        response_messages = []
        for msg in messages:
            sender = USERS.query.get(msg.sender_id)
            receiver = USERS.query.get(msg.receiver_id)
            
            response_messages.append({
                'sender': sender.username if sender else 'Unknown',
                'sender_name': sender.username if sender else 'Unknown',
                'text': msg.message
            })
        
        return jsonify({"messages": response_messages}), 200

    except Exception as e:
        print(f"Error fetching messages: {e}")
        return jsonify({"error": "Failed to fetch messages", "details": str(e)}), 500

@app.route('/persons_of_interest')
def persons_of_interest():
    persons = PersonOfInterest.query.all()
    return render_template('persons_of_interest.html', persons=persons)

@app.route('/add_person', methods=['POST'])
def add_person():
    name = request.form['name']
    alias = request.form.get('alias')
    description = request.form.get('description')
    last_seen_location = request.form.get('last_seen_location')
    user_id = session['user_id']
    
    notes = request.form.get('notes')

     # Handle file upload
    photo = request.files['photo']
    photo_path = None
    if photo and allowed_file(photo.filename):
        filename = secure_filename(photo.filename)
        photo_path = os.path.join(app.config['UPLOAD_FOLDER_POI'], filename)
        photo.save(photo_path)

    last_seen_date_str = request.form.get('last_seen_date')  # e.g., "2024-11-21T14:30"

# Convert to a datetime object
    if last_seen_date_str:
        last_seen_date = datetime.strptime(last_seen_date_str, "%Y-%m-%dT%H:%M")
        print(f"Converted datetime: {last_seen_date}")
    else:
        print("No date provided")
        
    
    new_person = PersonOfInterest(
        name=name,
        alias=alias,
        description=description,
        last_seen_location=last_seen_location,
        last_seen_date=last_seen_date,
        notes=notes,
        photo_path = photo_path,
        user_id=user_id
    )
    db.session.add(new_person)
    db.session.commit()
    return redirect(url_for('persons_of_interest'))

# Route for registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        mobile = request.form['mobile']
        access = 'inactive'  # Default access level for new users

        if USERS.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        if USERS.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = USERS(
            first_name=first_name,
            last_name=last_name,
            email=email,
            username=username,
            password=password,
            mobile=mobile,
            access=access,
            role="REPORTER"
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful as reporter! Please wait for activation before logging in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


# Route for login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query the user from the database
        user = USERS.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Successful login
            session['user_id'] = user.user_id
            session['username'] = user.username
            session['access'] = user.access
            session['role'] = user.role

            if user.role == 'ADMIN':
                return redirect(url_for('dashboard'))
            elif user.role == 'REPORTER':
                return redirect(url_for('home'))  # Redirect to reporter's home page
        else:
            # Invalid credentials or unauthorized role
            flash('Invalid username, password, or unauthorized access.', 'danger')
    
    # Render login page for GET requests or after a failed login
    return render_template('login.html')

@app.route('/session', methods=['GET'])
def get_session():
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user_id': session['user_id'],
            'username': session['username'],
            'role': session['role']
        })
    return jsonify({'logged_in': False})

# Route for logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Route to render the analysis form
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_data():
    if request.method == 'POST':
        # Get user inputs
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        location = request.form.get('location')

        # Query the database for incidents within the date range and location
        incidents = Incident.query.filter(
            Incident.timestamp >= datetime.strptime(start_date, '%Y-%m-%d'),
            Incident.timestamp <= datetime.strptime(end_date, '%Y-%m-%d'),
            Incident.location == location
        ).all()

        # Aggregate data for analysis
        reports = [incident.report_text for incident in incidents]
        combined_reports = " ".join(reports)
        data = combined_reports
        print(f"Data being sent to OpenAI: {data}")
        # Call OpenAI   for intelligent analysis
        openai_response = analyze_with_openai(combined_reports)

        return render_template('analysis_result.html', incidents=incidents, analysis=openai_response)

    return render_template('analysis_form.html')

def analyze_with_openai(data):
    """
    Use OpenAI to generate an intelligent analysis of the provided data.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent data analyst. Provide detailed analysis and actionable insights based on the user's input data."
                },
                {
                    "role": "user",
                    "content": f"Provide a detailed analysis and actionable insights for the following data:\n\n{data}"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error in OpenAI analysis: {e}"

#Sensor data from Packetworx
import dateutil.parser

@app.route('/api/sensor_data', methods=['POST'])
def create_sensor_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["sensor_id", "sensor_type", "value"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Parse the timestamp if provided, or use the current UTC time
        if "timestamp" in data:
            try:
                timestamp = dateutil.parser.parse(data["timestamp"])
            except ValueError:
                return jsonify({"error": "Invalid timestamp format. Use ISO 8601 format."}), 400
        else:
            timestamp = datetime.utcnow()

        new_sensor_data = SensorData(
            #sensor_id=data["sensor_id"],
            sensor_type=data["sensor_type"],
            value=data["value"],
            timestamp=timestamp,
            notes=data.get("notes"),
            other_data=data.get("other_data"),
            longitude=data.get("longitude"),
            latitude=data.get("latitude")
        )

        db.session.add(new_sensor_data)
        db.session.commit()

        return jsonify({"message": "Sensor data saved successfully"}), 201

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/view_sensor_data', methods=['GET'])
def view_sensor_data():
    # Query all the sensor data from the database
    sensor_data_records = SensorData.query.all()

    # Render the HTML template and pass the sensor_data
    return render_template('view_sensor_data.html', sensor_data=sensor_data_records)        


# Endpoint for IAQ
@app.route('/api/iaq', methods=['POST'])
def create_iaq():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Parse `actility_original_payload` and `decoded_payload`
        original_payload = data.get("actility_original_payload", {}).get("DevEUI_uplink", {})
        decoded_payload = data.get("uplink_message", {}).get("decoded_payload", {})

        # Parse timestamp or use the current time
        if "timestamp" in data:
            try:
                timestamp = dateutil.parser.parse(data["timestamp"])
            except ValueError:
                return jsonify({"error": "Invalid timestamp format. Use ISO 8601 format."}), 400
        else:
            timestamp = datetime.utcnow()

        # Extract fields from JSON
        new_iaq = IAQ(
            longitude=original_payload.get("LrrLON"),
            latitude=original_payload.get("LrrLAT"),
            device_id=data.get("end_device_ids", {}).get("device_id"),
            application_id=data.get("end_device_ids", {}).get("application_ids", {}).get("application_id"),
            received_at=data.get("received_at"),
            temperature=decoded_payload.get("temperature"),
            humidity=decoded_payload.get("humidity"),
            pir=decoded_payload.get("pir"),
            light_level=decoded_payload.get("light_level"),
            co2=decoded_payload.get("co2"),
            tvoc=decoded_payload.get("tvoc"),
            pressure=decoded_payload.get("pressure"),
            hcho=decoded_payload.get("hcho"),
            pm2_5=decoded_payload.get("pm2_5"),
            pm10=decoded_payload.get("pm10")
        )

        db.session.add(new_iaq)
        db.session.commit()

        return jsonify({"message": "IAQ data saved successfully"}), 201

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/oaq', methods=['POST'])
def create_oaq():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        
        # Extract top-level fields
        device_id = data["end_device_ids"].get("device_id")
        application_id = data["end_device_ids"]["application_ids"].get("application_id")
        #actility_data = data.get("actility_original_payload", {}).get("DevEUI_uplink", {})
        uplink_message = data.get("uplink_message", {})
        decoded_payload = uplink_message.get("decoded_payload", {})

        try:
            received_at = dateutil.parser.parse(data.get("received_at", datetime.utcnow().isoformat()))
        except ValueError:
            return jsonify({"error": "Invalid received_at format. Use ISO 8601 format."}), 400

        # Extract location and signal information from nested fields
        #lrrs = actility_data.get("Lrrs", {}).get("Lrr", [])
       # longitude = lrrs[0]["LrrLON"] if lrrs else None
        #latitude = lrrs[0]["LrrLAT"] if lrrs else None

        # Extract sensor readings from decoded payload
        raw_payload = decoded_payload.get("raw_payload")
        port = decoded_payload.get("port")
        bat_v = decoded_payload.get("bat_v")
        solarCharging = decoded_payload.get("solarCharging")
        solarCurrent = decoded_payload.get("solarCurrent")
        battCurrent = decoded_payload.get("battCurrent")
        solar_v = decoded_payload.get("solar_v")
        CO2 = decoded_payload.get("CO2")
        Hum = decoded_payload.get("Hum")
        Temp = decoded_payload.get("Temp")
        PM01 = decoded_payload.get("PM01")
        PM25 = decoded_payload.get("PM25")
        PM10 = decoded_payload.get("PM10")
        TVOC = decoded_payload.get("TVOC")
        NOX = decoded_payload.get("NOX")

        # Extract breach flags
        breachedHumMin = decoded_payload.get("breachedHumMin")
        breachedHumMax = decoded_payload.get("breachedHumMax")
        breachedTempMin = decoded_payload.get("breachedTempMin")
        breachedTempMax = decoded_payload.get("breachedTempMax")
        breachedCO2Min = decoded_payload.get("breachedCO2Min")
        #breachedCO2Max = decoded_payload.get("breachedCO2Max")
        breachedTVOCMax = decoded_payload.get("breachedTVOCMax")
        breachedNOXMax = decoded_payload.get("breachedNOXMax")

        # Create OAQ instance
        new_oaq = OAQ(
            device_id=device_id,
            application_id=application_id,
            received_at=received_at,
            raw_payload=raw_payload,
            port=port,
            bat_v=bat_v,
            solarCharging=solarCharging,
            solarCurrent=solarCurrent,
            battCurrent=battCurrent,
            solar_v=solar_v,
            CO2=CO2,
            Hum=Hum,
            Temp=Temp,
            PM01=PM01,
            PM25=PM25,
            PM10=PM10,
            TVOC=TVOC,
            NOX=NOX,
            breachedHumIn=breachedHumMin,
            #breachedHumMax=breachedHumMax,
            breachedTempMin=breachedTempMin,
            breachedTempMax=breachedTempMax,
            breachedCO2Min=breachedCO2Min,
            #breachedCO2Max=breachedCO2Max,
            #breachedTVOCMax=breachedTVOCMax,
            breachedNOXMax=breachedNOXMax
            #longitude=longitude,
           # latitude=latitude
        )

        # Save to database
        db.session.add(new_oaq)
        db.session.commit()

        return jsonify({"message": "OAQ data saved successfully"}), 201

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/noise', methods=['POST'])
def create_noise():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

         # Parse received_at or use the current time
        try:
            received_at = dateutil.parser.parse(data.get("received_at", datetime.utcnow().isoformat()))
        except ValueError:
            return jsonify({"error": "Invalid received_at format. Use ISO 8601 format."}), 400


        # Extract relevant fields
        decoded_payload = data.get("uplink_message", {}).get("decoded_payload", {})
        device_info = data.get("end_device_ids", {})
        location_info = data.get("actility_original_payload", {}).get("DevEUI_uplink", {})

        new_noise = Noise(
            device_id=device_info.get("device_id"),
            application_id=device_info.get("application_ids", {}).get("application_id"),
            received_at=received_at,
            raw_payload=decoded_payload.get("raw_payload"),
            batt_v=decoded_payload.get("batt_v"),
            averageNoise=decoded_payload.get("averageNoise"),
            notes=decoded_payload.get("noiseLevel"),
            maxNoise=decoded_payload.get("maxNoise"),
            MinNoise=decoded_payload.get("minNoise"),
            chargingVoltage=decoded_payload.get("chargingVoltage"),
            chargingCurrent=decoded_payload.get("chargingCurrent"),
            sensor_type=decoded_payload.get("chargingStatus"),
            transmission=decoded_payload.get("transmission"),
            longitude=location_info.get("LrrLON"),
            latitude=location_info.get("LrrLAT")
        )

        db.session.add(new_noise)
        db.session.commit()

        return jsonify({"message": "Noise data saved successfully"}), 201

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/testpayload', methods=['GET', 'POST'])
def payload():
    if request.method == 'POST':
        url = "https://demo.airalabs.ai/api/sensor_data"
        headers = {"Content-Type": "application/json"}
        payload = {
            "sensor_id": 101,
            "sensor_type": "temperature",
            "value": 25.5,
            "timestamp": "2025-01-25T15:30:00Z",
            "notes": "Outdoor sensor near greenhouse",
            "longitude": "120.5890",
            "latitude": "15.4210"
        }

        # Sending POST request to the target URL with the payload
        response = requests.post(url, json=payload, headers=headers)

        # Print status code and response
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json())  # Successful response handling
        else:
            print("Error: ", response.text)  # Handle error response

        return jsonify({"message": "Payload sent successfully!"}), 200

    # Handle GET request: Just return a simple message if GET is used
    return jsonify({"message": "Use POST method to send payload to /api/sensor_data"}), 200
    

@app.route('/analyze_incident/<int:incident_id>', methods=['GET', 'POST'])
def analyze_incident(incident_id):
    
    # Fetch the incident object from the database
    incident = Incident.query.get_or_404(incident_id)
    
    # Check if the request method is POST (form submission)
    if request.method == 'POST':
        # Extract action points and what to do from the form data
        action_points = request.form.get('action_points', '')
        
        
        # Extract tokens from the report_text (can be used for comparison)
        report_text = incident.report_text
        tokens = incident.tokens  # Tokens already exist in the incident object
        
        # Save the new analysis to the database
        analysis = IncidentAnalysis(
            incident_id=incident.id,
            action_points=action_points,
            report_text = report_text,
            tokens=tokens  # Tokens as comma-separated string
        )

        db.session.add(analysis)
        db.session.commit()

        # Now suggest action points based on similar incidents
    

    # Handle GET request (for displaying incident details and any suggested points)
    return render_template('analyze_incident.html', incident=incident)




@app.route('/dispatch', methods=['GET', 'POST'])
def dispatch():
    if request.method == 'POST':
        caller_name = request.form['caller_name']
        contact_number = request.form['contact_number']
        report_text = request.form['report_text']
        location = request.form['location']
        tag = request.form['tag']
        notes = request.form['notes']
        type ="dispatch"
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        complainant = request.form.get('complainant')
        defendant = request.form.get('defendant')

        category = categorize_incident(report_text)
        
    
        # Tokenization and analysis are performed here
        tokens = ", ".join([w for w in word_tokenize(report_text) if w.lower() not in stopwords.words('english')])
    
    # Detect the language of the report
        language = detect(report_text)

        # AI Analysis
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert dispatcher AI assistant."},
                {"role": "user", "content": f"Analyze this incident and suggest appropriate authorities or departments to dispatch: {report_text}"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        suggestions = openai_response['choices'][0]['message']['content'].strip()

        # Save incident to database
        new_incident = Incident(
            caller_name=caller_name,
            tag = tag,
            category=category,
            tokens=tokens,
            language = language,
            notes=notes,
            type = type,
            contact_number=contact_number,
            report_text=report_text,
            location=location,
            openai_analysis=suggestions,
            latitude = latitude,
            longitude=longitude,
            complainant = complainant,
            defendant = defendant,
            user_id = session['user_id'],
            timestamp=datetime.utcnow()
        )
        db.session.add(new_incident)
        db.session.commit()

        return redirect(url_for('dispatch'))

    incidents = Incident.query.order_by(Incident.timestamp.desc()).all()
    return render_template('dispatch.html', incidents=incidents)

@app.route('/send_dispatch/<int:incident_id>', methods=['POST'])
def send_dispatch(incident_id):
    incident = Incident.query.get_or_404(incident_id)
    authorities = request.form['assigned_authorities']
    incident.assigned_authorities = authorities
    db.session.commit()
    return jsonify({'message': 'Dispatch sent successfully!'})

@app.route('/ai', methods=['GET', 'POST'])
def ai_page():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    # Default filter: incidents from today and yesterday
    filter_conditions = []
    
    # Collecting search query from the form (location, category, and query)
    location_filter = request.form.get('location')
    category_filter = request.form.get('category')
    query_filter = request.form.get('query')

    group_by = request.args.get('group_by', 'tokens')  # Default to 'tokens'
    page = request.args.get('page', default=1, type=int)
    total_pages = 10

    # Building the filter conditions based on the provided filters
    if request.method == 'POST':
        if location_filter:
            filter_conditions.append(Incident.location == location_filter)
        if category_filter:
            filter_conditions.append(Incident.category == category_filter)
    
    # Fetch incidents based on filter conditions
    incidents = fetch_incidents(filter_conditions)

    if not incidents:
        return render_template('ai.html', 
                               location_stats=[], 
                               category_stats=[], 
                               crime_prediction=[],
                               location_fig_html='',
                               category_fig_html='',
                               group_by = group_by,
                               page=page,
                               total_pages= total_pages,
                               query_result='No incidents found for the selected filters.')

    # Analyze the incidents to generate stats
    location_stats, category_stats, date_stats = analyze_incidents(incidents)
    
    # Create visualizations for incidents by location and category
    fig_location = px.bar(location_stats, x='location', y='incident_count', title="Incidents by Location")
    fig_category = px.bar(category_stats, x='category', y='incident_count', title="Incidents by Category")
    
    # Generate HTML for the charts
    location_fig_html = fig_location.to_html(full_html=False)
    category_fig_html = fig_category.to_html(full_html=False)
    
    # Predict incidents based on the filtered incidents
    incident_prediction = predict_incidents(incidents)
    session['incident_prediction'] = incident_prediction

    # Handle the natural language query if provided
    query_result = ''
    if query_filter:
        query_result = handle_natural_language_query(query_filter)

    # Render the template with data and charts
    return render_template('ai.html', 
                           location_stats=location_stats.to_dict(orient='records'),
                           category_stats=category_stats.to_dict(orient='records'),
                           incident_prediction=incident_prediction,
                           location_fig_html=location_fig_html, 
                           category_fig_html=category_fig_html,
                           group_by=group_by,
                           page=page,
                           total_pages = total_pages,
                           query_result=query_result)



@app.route('/update_tag/<incident_id>', methods=['POST'])
def update_tag(incident_id):
    tag = request.form['tag']
    incident = Incident.query.get(incident_id)
    if incident:
        incident.tag = tag
        db.session.commit()

    return redirect(url_for('incident_details', incident_id=incident_id))

@app.route('/add_response/<int:incident_id>', methods=['POST'])
def add_response(incident_id):
    user_id = session['user_id']  # Replace with actual user ID from session or authentication
    response_text = request.form.get('response')
    timestamp = datetime.now()
    tag = request.form['tag']

    new_response = ResponseDB(
        incident_id=incident_id,
        user_id=user_id,
        response=response_text,
        timestamp=timestamp,
        tag=tag
    )
    db.session.add(new_response)
    db.session.commit()

    incident = Incident.query.get(incident_id)
    if incident:
        incident.tag = tag  # Update the incident's tag
        db.session.commit()

    return redirect(url_for('incident_details', incident_id=incident_id))

@app.route('/map')
def map_page():
    return render_template('map.html')  # This will render the map.html template


@app.route('/map_data', methods=['GET'])
def map_data():
    # Get the 'start_date', 'end_date' and 'search_query' query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    search_query = request.args.get('query', '').lower()  # New parameter for search query
    print(search_query)
    # Default to today's date if no date range is provided
    if not start_date or not end_date:
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = end_date = today

    # Convert string dates to datetime objects (with default time)
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Set the time for start_date to 00:00:00 and end_date to 23:59:59
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Query incidents from the database and filter by date range
    incidents = Incident.query.filter(Incident.timestamp >= start_date, Incident.timestamp <= end_date)

    if search_query:
        # If there's a natural language search query, filter based on report_text or category
        doc = nlp(search_query)
        
        # Extract keywords (nouns and proper nouns) and named entities from the query
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        entities = [ent.text for ent in doc.ents]
        
        # Combine keywords and entities for search criteria (remove duplicates)
        search_terms = set(keywords + entities)
        
        # Remove stopwords from search terms
        stop_words = set(stopwords.words('english'))
        filtered_search_terms = [term for term in search_terms if term.lower() not in stop_words]

        # Query the database with the filtered search terms using TF-IDF and Cosine Similarity
        if filtered_search_terms:
            # Extract report texts from all incidents
            all_incidents = Incident.query.all()
            all_reports = [i.report_text for i in all_incidents]

            # Vectorize the report text and the search query
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_reports + [' '.join(filtered_search_terms)])
            
            # Compute cosine similarity between the query and the incident reports
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            
            # Sort incidents by similarity to the search query
            similar_indices = cosine_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar incidents
            incidents = [all_incidents[i] for i in similar_indices]

            print(f"THERE ARE INCIDENTS: {incidents}")

    # Convert the query results to a list of incident details
    filtered_incidents = [
        {
            'id': incident.id,
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'category': incident.category,
            'location': incident.location,
            'report_text': incident.report_text,
            'media_path': incident.media_path,
            'openai_analysis': incident.openai_analysis,
            'actionpoints': incident.actionpoints,
            'recommendation': incident.recommendation,
            'crops_affected': incident.crops_affected,
            'damage_estimate': incident.damage_estimate,
            'field_notes': incident.field_notes,
            'timestamp': incident.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        for incident in incidents
    ]

    app.logger.debug(f"Filtered Map Data from {start_date} to {end_date} with search query '{search_query}': {filtered_incidents}")

    return jsonify(filtered_incidents)






def group_related_incidents(incidents):
    """Groups incidents by token, category, and report_text similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    grouped = {}

    for incident in incidents:
        # Create a string key for grouping
        key = f"{incident.tokens}-{incident.category}"
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(incident)

    # Further group by text similarity within each key
    for key, incidents_list in grouped.items():
        vectorizer = TfidfVectorizer()
        texts = [incident.report_text for incident in incidents_list]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Add advanced grouping logic if needed here

    return grouped



def generate_insights(grouped_incidents):
    """Generates insights for each group."""
    insights = []
    for key, incidents in grouped_incidents.items():
        # Handle the case where tokens or category have commas or special characters
        # Split the key by a safe delimiter like "|" and avoid multiple delimiters
        key_parts = re.split(r"[,|-]+", key)
        
        if len(key_parts) < 2:
            raise ValueError(f"Unexpected key format: {key}")
        
        tokens = key_parts[0]
        category = key_parts[-1]
        
        # Generate insights for the group
        urgent_count = sum(1 for i in incidents if i.category.lower() == 'urgent')
        locations = {i.location for i in incidents}
        insights.append({
            "tokens": tokens,
            "category": category,
            "urgent_cases": urgent_count,
            "locations": list(locations),
            "insight": f"{len(locations)} unique locations with {urgent_count} urgent cases."
        })
    return insights

@app.route('/top_incidents', methods=['GET'])
def top_incidents():
    # Group incidents by location, category, and timestamp
    incidents = Incident.query.all()
    
    # Aggregation
    location_counts = {}
    category_counts = {}
    date_counts = {}

    for incident in incidents:
        location = incident.location
        category = incident.category
        date = incident.timestamp.date()
        
        location_counts[location] = location_counts.get(location, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        date_counts[date] = date_counts.get(date, 0) + 1
    
    # Sort by frequency
    sorted_location_counts = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_category_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_date_counts = sorted(date_counts.items(), key=lambda x: x[1], reverse=True)

    return render_template('top_incidents.html', 
                           location_counts=sorted_location_counts,
                           category_counts=sorted_category_counts,
                           date_counts=sorted_date_counts)




#GENERATE REPORTS FOR REPORT.HTML
from collections import defaultdict
from datetime import datetime, timedelta
from flask import request, render_template
from sqlalchemy.orm import joinedload
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime
# Download stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize sentence transformer model (supports multilingual, including Tagalog)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Define a list of Tagalog stopwords (can be extended further)
tagalog_stopwords = [
    'ang', 'ng', 'sa', 'at', 'ay', 'ako', 'ikaw', 'siya', 'kami', 'sila', 'ito', 'iyan', 'iyon', 'mga', 'naman', 
    'kasi', 'kung', 'saan', 'paano', 'kaya', 'bawat', 'lahat', 'wala', 'may', 'para', 'dahil', 'dapat', 
    'na'
]

# Combine English and Tagalog stopwords
stop_words = set(stopwords.words('english')).union(set(tagalog_stopwords))

def remove_stopwords(text):
    """Remove stopwords from the text."""
    text = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]  # Ignore short words
    return " ".join(filtered_words)

def encode_text(text):
    """Encode the text using sentence transformer to get embeddings."""
    return np.array(model.encode([text]))  # Ensure the output is a numpy array for cosine similarity

@app.route('/reports', methods=['GET'])
def reports():
    """
    Generate and render incident reports with advanced search and natural language processing.
    """
    # Get filter inputs

    incident_id = request.args.get('id')
    category = request.args.get('category')
    location = request.args.get('location')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    report_text = request.args.get('report_text')
    search_query = request.args.get('nl_search')  # Natural language search query
    incident_type = request.args.get('type')
    tag = request.args.get('tag')

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    # Base query
    query = Incident.query

    # Apply advanced filters
    if incident_id:
        query = query.filter(Incident.id == incident_id)
    if category:
        query = query.filter(Incident.category.ilike(f"%{category}%"))
    if location:
        query = query.filter(Incident.location.ilike(f"%{location}%"))
    if start_date:
        query = query.filter(Incident.timestamp >= start_date)
    if end_date:
        query = query.filter(Incident.timestamp <= end_date)
    if report_text:
        query = query.filter(Incident.report_text.ilike(f"%{report_text}%"))
    if incident_type:
        query = query.filter(Incident.type == incident_type)
    if tag:
        query = query.filter(Incident.tag == tag)

    # Exclude 'blotter' type records
    query = query.filter(Incident.type != 'blotter')

    # Apply natural language search
    if search_query:
        import spacy
        from nltk.corpus import stopwords
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Load NLP model
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(search_query)

        # Extract keywords and entities
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        entities = [ent.text for ent in doc.ents]

        # Combine keywords and entities, remove duplicates and stopwords
        search_terms = set(keywords + entities)
        stop_words = set(stopwords.words('english'))
        filtered_search_terms = [term for term in search_terms if term.lower() not in stop_words]

        # Query the database with the filtered search terms
        if filtered_search_terms:
            all_incidents = query.all()  # Get all incidents based on previous filters
            all_reports = [i.report_text for i in all_incidents]

            # Use TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_reports + [' '.join(filtered_search_terms)])

            # Compute cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

            # Sort incidents by similarity
            similar_indices = cosine_sim.argsort()[0][-5:][::-1]  # Top 5 most similar
            filtered_incidents = [all_incidents[i] for i in similar_indices]
        else:
            filtered_incidents = []
    else:
        filtered_incidents = query.order_by(Incident.timestamp.desc()).paginate(
            page=page, per_page=per_page
        ).items

    # Pagination
    if search_query:
        total_filtered = len(filtered_incidents)
        paginated_incidents = filtered_incidents[(page - 1) * per_page: page * per_page]
        pagination = {
            'total': total_filtered,
            'page': page,
            'per_page': per_page,
            'pages': (total_filtered // per_page) + (1 if total_filtered % per_page > 0 else 0),
        }
    else:
        paginated_query = query.paginate(page=page, per_page=per_page)
        paginated_incidents = paginated_query.items
        pagination = {
            'total': paginated_query.total,
            'page': paginated_query.page,
            'per_page': paginated_query.per_page,
            'pages': paginated_query.pages,
        }

    # Render the template with results
    return render_template(
        'reports.html',
        incidents=paginated_incidents,
        pagination=pagination,
        id=incident_id,
        category=category,
        location=location,
        start_date=start_date,
        end_date=end_date,
        report_text=report_text,
        nl_search=search_query,
        type=incident_type,
        tag=tag,
        per_page=per_page
    )

@app.route('/blotter_reports', methods=['GET'])
def blotter_reports():
    """
    Generate and render blotter reports.
    """
    # Pagination parameters
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=10, type=int)

    # Get blotter incidents from the database
    blotter_incidents = Incident.query.filter(Incident.type == 'blotter').order_by(Incident.timestamp.desc()).all()

    # Pagination logic
    pagination = None
    if blotter_incidents:
        pagination = {
            'total': len(blotter_incidents),
            'page': page,
            'per_page': per_page,
            'pages': (len(blotter_incidents) // per_page) + 1,
            'items': blotter_incidents[(page - 1) * per_page: page * per_page]
        }

    # Render the template with results
    return render_template(
        'blotter_reports.html',
        incidents=pagination['items'] if pagination else blotter_incidents,
        pagination=pagination
    )

@app.route('/search', methods=['GET'])
def search():
    try:
        search_query = request.args.get('search_query', '')
        
        # Default to empty list if no search query is provided
        incidents = []
        pagination = None  # Initialize pagination variable
        
        if search_query:
            # Preprocess the search query using spaCy for NER and tokenization
            doc = nlp(search_query)
            
            # Extract keywords (nouns and proper nouns) and named entities from the query
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            entities = [ent.text for ent in doc.ents]
            
            # Combine keywords and entities for search criteria (remove duplicates)
            search_terms = set(keywords + entities)
            
            # Remove stopwords from search terms
            stop_words = set(stopwords.words('english'))
            filtered_search_terms = [term for term in search_terms if term.lower() not in stop_words]

            # Query the database with the filtered search terms using TF-IDF and Cosine Similarity
            if filtered_search_terms:
                # Extract report texts from all incidents
                all_incidents = Incident.query.all()
                all_reports = [i.report_text for i in all_incidents]

                # Vectorize the report text and the search query
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_reports + [' '.join(filtered_search_terms)])
                
                # Compute cosine similarity between the query and the incident reports
                cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                
                # Sort incidents by similarity to the search query
                similar_indices = cosine_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar incidents
                incidents = [all_incidents[i] for i in similar_indices]

        # Add pagination for the search results
        page = request.args.get('page', 1, type=int)  # Get the page number from query string
        per_page = 10  # Define how many results per page
        pagination = Pagination(page=page, total=len(incidents), per_page=per_page, record_name='incidents')

        # Paginate the incidents
        paginated_incidents = incidents[(page - 1) * per_page: page * per_page]

        # Safeguard: Ensure pagination object is not None and has valid attributes
        start_page = 1
        end_page = pagination.pages if pagination else 1
        if pagination and pagination.page:
            start_page = max(pagination.page - 2, 1)
            end_page = min(pagination.page + 2, pagination.pages)

        # Render the results in the dashboard template, passing pagination data
        return render_template(
            'dashboard.html',
            incidents=paginated_incidents,
            pagination=pagination,
            start_page=start_page,
            end_page=end_page,
            search_query=search_query  # Pass the search query to the template
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to get similar report texts using cosine similarity
def get_similar_reports2(input_report_text, analysis_records):
    # Ensure the input report_text is valid
    if not input_report_text:
        raise ValueError("Input report text is empty or invalid")

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Combine the input report text and the report texts from the analysis records
    text_data = [input_report_text] + [record.report_text for record in analysis_records if record.report_text]

    if not text_data:
        raise ValueError("No valid report texts provided for analysis")
    
    # Fit and transform the text data
    text_matrix = vectorizer.fit_transform(text_data)

    # Compute similarity between the input report text and each analysis record
    similarity_scores = cosine_similarity(text_matrix[0:1], text_matrix[1:])
    
    return similarity_scores


from flask_caching import Cache

# Initialize cache globally
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
# Configure the cache (simple cache for demonstration)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Initialize the Cache with the app
cache = Cache(app)

@cache.cached(timeout=300, key_prefix='incident_analysis')
def get_incident_analysis():
    """Fetch all analysis records from the database with caching."""
    return IncidentAnalysis.query.all()

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer, util
import spacy



# Load the pre-trained SentenceTransformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the spacy model for NLP preprocessing
nlp = spacy.load('en_core_web_sm')

# Define additional Tagalog stopwords
TAGALOG_STOPWORDS = {
    "ang", "mga", "sa", "ng", "ako", "ikaw", "siya", "sila", "kami", "kayo", 
    "ito", "iyon", "ay", "at", "na", "pero", "kung", "dahil", "para", "po", 
    "ho", "dito", "doon", "iyan", "may", "wala", "meron", "paano", "ngayon", 
    "noon", "bakit", "lahat", "hindi", "oo", "o", "nga", "naman", "pa", "ba",
    "lang", "ganun", "ganito", "diba", "eh", "kasi", "talaga"
}

def preprocess_text(text, stop_words, disregard_words):
    """
    Preprocesses text by removing stopwords, disregard words, and non-alphabetical tokens.
    
    Args:
        text (str): The input text to preprocess.
        stop_words (set): A set of stopwords to filter out.
        disregard_words (set): A set of words to disregard during matching.
    
    Returns:
        str: The preprocessed text.
    """
    doc = nlp(text)
    filtered_tokens = [
        token.lemma_ for token in doc 
        if token.is_alpha and token.text.lower() not in stop_words 
        and token.text.lower() not in disregard_words and not token.ent_type_
    ]
    return ' '.join(filtered_tokens)

def extract_keywords(text, stop_words, disregard_words):
    """
    Extracts significant keywords from a text, excluding stopwords and disregard words.
    """
    doc = nlp(text)
    return {
        token.text.lower() for token in doc 
        if token.is_alpha 
        and token.text.lower() not in stop_words 
        and token.text.lower() not in disregard_words  # Exclude disregard words
        and not token.ent_type_
    }

def keyword_filter(text1, text2, stop_words, disregard_words):
    """
    Checks for keyword overlap between two texts to verify semantic match, ignoring disregard words.
    
    Args:
        text1 (str): First text input.
        text2 (str): Second text input.
        stop_words (set): Stopwords to filter out.
        disregard_words (set): Words to disregard in keyword matching.
    
    Returns:
        bool: True if there is significant overlap in key terms, otherwise False.
    """
    keywords1 = extract_keywords(text1, stop_words, disregard_words)
    keywords2 = extract_keywords(text2, stop_words, disregard_words)
    
    # Check for intersection of key terms
    common_keywords = keywords1.intersection(keywords2)
    return len(common_keywords) > 0

def get_similar_reports(report_text, analysis_records, incident, threshold=0.7):
    """
    Finds similar reports using a hybrid approach: Sentence Transformers and keyword-based matching.
    
    Args:
        report_text (str): The text of the current incident report.
        analysis_records (list): A list of incident analysis records.
        incident (dict): The incident object containing disregard_words.
        threshold (float): Similarity score threshold for matching.
    
    Returns:
        list: A list of similar records with action points and scores.
    """
    if analysis_records is None:
        print("Error: analysis_records is None!")
        return []

    if not analysis_records:
        print("Debug: No analysis records found (empty list).")
        return []

    # Merge default stopwords with Tagalog-specific stopwords
    stop_words = set(nlp.Defaults.stop_words).union(TAGALOG_STOPWORDS)
    
    # Get disregard words from the incident object
    disregard_words = set(incident.disregard_words) if hasattr(incident, 'disregard_words') else set()
    print(f"Debug: Disregard words: {disregard_words}")

    # Preprocess the report text and analysis texts
    processed_report_text = preprocess_text(report_text, stop_words, disregard_words)
    
    # For each analysis record, preprocess and compute similarity
    similar_reports = []
    for record in analysis_records:
        if 'report_text' not in record:
            print(f"Error: Missing 'report_text' in record {record}")
            continue

        processed_analysis_text = preprocess_text(record['report_text'], stop_words, disregard_words)

        # Encode the processed texts using sentence transformer model
        query_embedding = model.encode([processed_report_text], convert_to_tensor=True)
        analysis_embedding = model.encode([processed_analysis_text], convert_to_tensor=True)

        # Compute cosine similarity
        cosine_score = util.pytorch_cos_sim(query_embedding, analysis_embedding)[0][0].item()

        # Extract keywords for comparison
        keywords = extract_keywords(report_text, stop_words, disregard_words)
        record_keywords = extract_keywords(record['report_text'], stop_words, disregard_words)

        # Debug keyword overlap
        print(f"Debug: Keywords1 = {keywords}, Keywords2 = {record_keywords}")

        # Filter based on either cosine similarity or keyword match
        if cosine_score >= threshold or keyword_filter(report_text, record['report_text'], stop_words, disregard_words):
            similar_reports.append({
                "text": record['report_text'],
                "action_points": record.get('action_points', ''),
                "score": cosine_score,
                "keyword_overlap": list(keywords.intersection(record_keywords))  # Optional: for debugging keyword matches
            })

    # Sort reports by similarity score (descending order)
    similar_reports.sort(key=lambda x: x['score'], reverse=True)
    
    return similar_reports

def get_incident_analysis():
    return IncidentAnalysis.query.all()  # This will return ORM objects

@app.route('/get_survey_questions/<int:survey_id>', methods=['GET'])
def get_survey_questions(survey_id):
    survey = Survey.query.get(survey_id)
    if not survey:
        return jsonify({'error': 'Survey not found'}), 404

    questions = []
    for question in survey.questions:
        question_data = {
            'id': question.id,
            'text': question.text,
            'question_type': question.question_type,  # Ensure this field exists in your model
        }
        
        # Include options if the question type is MULTIPLE_CHOICE
        if question.question_type == 'MULTIPLE_CHOICE':
            question_data['options'] = [option.text for option in question.options]
        
        questions.append(question_data)

    return jsonify({'questions': questions})

@app.route('/incident/<int:incident_id>', methods=['GET', 'POST'])
def incident_details(incident_id):
    try:
        # Query the incident by ID
        incident = Incident.query.get_or_404(incident_id)
        user = USERS.query.filter_by(user_id=incident.user_id).first()
        user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
        markers =Marker.query.all()
        nearby_markers = get_nearby_markers(incident.latitude, incident.longitude)


        # Fetch all incident analysis records and convert them to dictionaries
        incident_analysis_records = get_incident_analysis()
        incident_analysis_dicts = [analysis.to_dict() for analysis in incident_analysis_records]

        # Find similar reports using the optimized function
        similar_reports = get_similar_reports(report_text=incident.report_text, 
                                            analysis_records=incident_analysis_dicts,
                                            incident=incident,
                                            threshold=0.85)  # Adjusted threshold for higher accuracy

# Collect unique action points, ensuring no duplicates
        action_points = set()

# Check for exact matches first
        exact_matches = [report for report in incident_analysis_dicts if report['report_text'] == incident.report_text]

# Add action points from exact matches
        for report in exact_matches:
            if 'action_points' in report and report['action_points']:
                points = report['action_points'].split('\n')  # Assuming action points are newline-separated
                action_points.update([point.strip() for point in points if point.strip()])

# Add action points from similar reports
        for report in similar_reports:
            if 'action_points' in report and report['action_points']:
                points = report['action_points'].split('\n')
                action_points.update([point.strip() for point in points if point.strip()])

# Convert the set back to a sorted list for rendering
        action_points = sorted(action_points)

        # SIMILAR INCIDENTS LIST
        # Fetch all incidents for comparison
        all_incidents = Incident.query.all()

        # Extract report texts from all incidents
        all_reports = [i.report_text for i in all_incidents]

        # Vectorize the report text using TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_reports)

        # Get the index of the current incident in the list
        current_index = all_incidents.index(incident)

        # Calculate cosine similarity between the current incident and all others
        cosine_sim = cosine_similarity(tfidf_matrix[current_index], tfidf_matrix)

        # Get the most similar incidents based on cosine similarity
        similar_indices = cosine_sim.argsort()[0][-6:-1]  # Get top 5 similar incidents (excluding itself)

        # Prepare the data to be passed to the template
        similar_incidents = [all_incidents[i] for i in similar_indices]

        # Prepare the data for displaying the current incident
        incident_data = {
            'id': incident.id,
            'report_text': incident.report_text,
            'timestamp': incident.timestamp,
            'category': incident.category,
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'tokens': incident.tokens,
            'openai_analysis': incident.openai_analysis,
            'user_id': incident.user_id,
            'location': incident.location,
            'recommendation': incident.recommendation,
            'language': incident.language,
            'field_notes': incident.field_notes,
            'crops_affected': incident.crops_affected,
            'damage_estimate': incident.damage_estimate,

            'media_path': incident.media_path if incident.media_path else None
        }

        # Fetch responses for this incident, ordered by timestamp descending
        responses = ResponseDB.query.filter_by(incident_id=incident_id).order_by(ResponseDB.timestamp.desc()).all()
       
        matches = None
       # if incident.media_path:
           # matches = analyze_media(incident.media_path)

        # Handle adding a new response
        if request.method == 'POST':
            response_text = request.form['response']
            new_response = Response(
                user_id=session['user_id'],  # Assuming you have a current_user for logged-in user
                response=response_text,
                incident_id=incident.id,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_response)
            db.session.commit()
            return redirect(url_for('incident_details', incident_id=incident_id))

        return render_template('incident_details.html', incident=incident_data, 
                               similar_incidents=similar_incidents, responses=responses, 
                               action_points=action_points, matches=matches, user_name=user_name,nearby_markers=nearby_markers)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from math import radians, sin, cos, sqrt, atan2

# Haversine formula to calculate the distance between two lat/lng points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in kilometers

def get_nearby_markers(incident_lat, incident_lon, radius_km=2):
    markers = Marker.query.all()  # Query markers from the database
    nearby_markers = []

    # Define a mapping from categories to icon filenames
    category_icons = {
        'Flood Prone Area': 'flood.png',
        'Fire Prone Area': 'fire.png',
        'Landslide Area': 'landslide.png',
        # Add more categories and icons as needed
    }

    for marker in markers:
        distance = haversine(incident_lat, incident_lon, marker.latitude, marker.longitude)
        if distance <= radius_km:
            marker.distance = distance  # Attach distance to marker

            # Set icon based on marker category
            icon_filename = category_icons.get(marker.category, 'default.png')  # Default icon if category not found

            nearby_markers.append({
                'latitude': marker.latitude,
                'longitude': marker.longitude,
                'label': marker.label,
                'icon': icon_filename, # Send only the icon filename to the front-end
                'distance':distance
            
            })

    return nearby_markers

def analyze_media(media_path):
    folder_paths = ['./static/uploads', './static/photos/POI']  # Folders to search for matches
    matches = []

    # Helper function to calculate similarity
    def calculate_similarity(img1, img2):
        diff = cv2.absdiff(img1, img2)
        return np.mean(diff)  # Lower scores indicate higher similarity

    # Helper function for face detection and matching
    def detect_and_match_faces(target_img, compare_img, media_type='photo'):
        # Load the pre-trained face detector (Haar Cascade or a deep learning model)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert images to grayscale for face detection
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        compare_gray = cv2.cvtColor(compare_img, cv2.COLOR_BGR2GRAY)

        # Detect faces in both images
        target_faces = face_cascade.detectMultiScale(target_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        compare_faces = face_cascade.detectMultiScale(compare_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If faces are detected, extract the face region and compare
        if len(target_faces) > 0 and len(compare_faces) > 0:
            for (x, y, w, h) in target_faces:
                target_face = target_img[y:y+h, x:x+w]
                for (cx, cy, cw, ch) in compare_faces:
                    compare_face = compare_img[cy:cy+ch, cx:cx+cw]
                    
                    # Resize both faces for consistent comparison
                    target_face_resized = cv2.resize(target_face, (256, 256), interpolation=cv2.INTER_AREA)
                    compare_face_resized = cv2.resize(compare_face, (256, 256), interpolation=cv2.INTER_AREA)

                    # Calculate similarity score between faces
                    score = calculate_similarity(target_face_resized, compare_face_resized)

                    if score < 20:  # Threshold for similarity
                        return True
        return False

    # Helper function to search a folder for matches
    def search_folder(folder, target_img=None, media_type='photo'):
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if media_type == 'photo' and file_name.endswith(('.jpg', '.jpeg', '.png')):
                compare_img = cv2.imread(file_path)
                if compare_img is None:
                    print(f"Failed to read image: {file_path}")
                    continue
                
                if detect_and_match_faces(target_img, compare_img):
                    matches.append({
                        'type': 'photo',
                        'file_name': file_name,
                        'link': url_for('static', filename=os.path.relpath(file_path, './static').replace('\\', '/'))
                    })

            elif media_type == 'video' and file_name.endswith('.mp4'):
                compare_cap = cv2.VideoCapture(file_path)
                compare_frame_count = 0
                while True:
                    ret_compare, compare_frame = compare_cap.read()
                    if not ret_compare:
                        break
                    if compare_frame_count % 30 == 0:
                        compare_frame_gray = cv2.cvtColor(compare_frame, cv2.COLOR_BGR2GRAY)
                        if detect_and_match_faces(target_img, compare_frame_gray):
                            matches.append({
                                'type': 'video',
                                'file_name': file_name,
                                'link': url_for('static', filename=os.path.relpath(file_path, './static').replace('\\', '/'))
                            })
                    compare_frame_count += 1
                compare_cap.release()

    # Analyze photo
    if media_path.endswith(('.jpg', '.jpeg', '.png')):
        target_img_path = media_path
        print(f"Target image path: {target_img_path}")
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            print(f"Error: Could not read target image {target_img_path}")
            return matches

        for folder in folder_paths:
            if os.path.exists(folder):
                print(f"Searching in folder: {folder}")
                search_folder(folder, target_img=target_img, media_type='photo')
            else:
                print(f"Folder does not exist: {folder}")

    # Analyze video
    elif media_path.endswith('.mp4'):
        target_video_path = os.path.join('./static', media_path)
        print(f"Target video path: {target_video_path}")
        cap = cv2.VideoCapture(target_video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for folder in folder_paths:
                    if os.path.exists(folder):
                        search_folder(folder, target_img=frame_gray, media_type='video')
            frame_count += 1
        cap.release()

    print(f"Matches found: {matches}")
    return matches




from scipy.spatial.distance import euclidean
from mtcnn import MTCNN  # MTCNN for face detection
import dlib
import cv2
import os
import numpy as np
from flask import jsonify

# Initialize MTCNN detector and Dlib's face recognition model
detector = MTCNN()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to detect faces using MTCNN
def detect_faces_mtcnn(image):
    if image is None or len(image.shape) < 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a valid RGB image.")
    detections = detector.detect_faces(image)
    return [(d['box'][0], d['box'][1], d['box'][2], d['box'][3]) for d in detections]

# Function to get normalized face embeddings
def get_face_embedding(image, face_rect):
    rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3])
    shape = sp(image, rect)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor) / np.linalg.norm(face_descriptor)

# Function to calculate similarity using Euclidean distance
def calculate_face_similarity(embedding1, embedding2):
    return euclidean(embedding1, embedding2)

# Function to compare two images
def compare_images(img1, img2):
    faces1 = detect_faces_mtcnn(img1)
    faces2 = detect_faces_mtcnn(img2)

    if not faces1 or not faces2:
        return None  # No faces detected in either image

    embeddings1 = [get_face_embedding(img1, rect) for rect in faces1]
    embeddings2 = [get_face_embedding(img2, rect) for rect in faces2]

    similarities = [calculate_face_similarity(emb1, emb2) for emb1 in embeddings1 for emb2 in embeddings2]
    # Return the best (min) similarity score
    return min(similarities, default=None)

# Function to analyze media (photos or videos)
def analyze_media2(media, media_type='photo'):
    folder_paths = ['./static/uploads', './static/photos/POI']
    matches = []

    if media_type == 'photo':
        for folder_path in folder_paths:
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img_to_compare = cv2.imread(img_path)

                if img_to_compare is None or len(img_to_compare.shape) < 3 or img_to_compare.shape[2] != 3:
                    continue

                similarity_score = compare_images(media, img_to_compare)
                if similarity_score is not None:  # Include all matches, even similarity score 0.0
                    matches.append({'filename': filename, 'similarity': similarity_score})

    elif media_type == 'video':
        video_capture = cv2.VideoCapture(media)
        frame_count = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % 30 == 0:  # Process every 30th frame
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except:
                    continue

                for folder_path in folder_paths:
                    for filename in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, filename)
                        img_to_compare = cv2.imread(img_path)

                        if img_to_compare is None or len(img_to_compare.shape) < 3 or img_to_compare.shape[2] != 3:
                            continue

                        similarity_score = compare_images(frame, img_to_compare)
                        if similarity_score is not None:  # Include all matches
                            matches.append({'filename': filename, 'similarity': similarity_score})

            frame_count += 1
        video_capture.release()

    # Filter matches to only include similarity score <= 0.3 (or 0.0)
    print("Matches before filtering:", matches)
    matches = [match for match in matches if match['similarity'] <= 0.45]
    print("Matches after filtering:", matches)

    # Sort matches by similarity in ascending order (lower score = better .match)
    matches = sorted(matches, key=lambda x: x['similarity'])

    return matches
from io import BytesIO
import tempfile

from flask import render_template, request, flash

# Store progress globally (this could be session or a more sophisticated solution)
processing_progress = 0

@app.route('/upload_media', methods=['GET', 'POST'])
def upload_media():
    global processing_progress
    matches = []
    
    # Reset progress when a new request is received
    processing_progress = 0

    if request.method == 'POST':
        photo = request.files.get('photo')
        video = request.files.get('video')

        # Process photo if uploaded
        if photo and allowed_file(photo.filename):
            photo_data = photo.read()  # Read photo into memory
            
            if not photo_data:
                flash('Error: Empty photo file.', 'danger')
            else:
                photo_img = cv2.imdecode(np.frombuffer(photo_data, np.uint8), cv2.IMREAD_COLOR)
                
                if photo_img is None or len(photo_img.shape) != 3 or photo_img.shape[2] != 3:
                    flash('Error: Invalid photo format. Please upload a valid color image.', 'danger')
                else:
                    matches = analyze_media2(photo_img, media_type='photo')
                    print(f"Matches Data: {matches}")  # Debug print for matches

                    # Simulate processing progress
                    for i in range(10):
                        time.sleep(0.2)  # Simulate some processing time
                        processing_progress = i * 10  # Update progress

        # Process video if uploaded
        if video and allowed_file(video.filename):
            video_data = video.read()  # Read video into memory
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_data)
                tmp_video_path = tmp_video.name

            try:
                video_cap = cv2.VideoCapture(tmp_video_path)
                if not video_cap.isOpened():
                    flash('Error: Couldn\'t open the video.', 'danger')
                else:
                    matches = analyze_media2(video_cap, media_type='video')
                    print(f"Matches Data: {matches}")  # Debug print for matches

                    # Simulate processing progress
                    for i in range(10):
                        time.sleep(0.2)  # Simulate some processing time
                        processing_progress = i * 10  # Update progress
            finally:
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)

        # Filter out matches with similarity 0.0 but still show them
        matches = [match for match in matches if match['similarity'] > 0.0 or match['similarity'] == 0.0]

        # Sort matches by similarity in ascending order (lower similarity = better match)
        matches = sorted(matches, key=lambda x: x['similarity'])

        # Construct `link` for each match
        folder_path = '/static/uploads/'  # Default folder for uploads
        POI_folder_path = os.path.join(app.root_path, 'static/photos/POI/')  # POI folder path on the server

        for match in matches:
            filename = match.get('filename')
            print(f"Checking for file: {filename}")  # Debug print for filenames
            
            if filename:
                full_POI_path = os.path.join(POI_folder_path, filename)

                # Check if the file exists in POI folder
                if os.path.exists(full_POI_path):
                    print(f"Found {filename} in POI folder.")
                    match['link'] = f"/static/photos/POI/{filename}"  # Use POI folder path for link
                else:
                    print(f"{filename} NOT found in POI folder.")
                    match['link'] = f"{folder_path}{filename}"  # Use regular uploads folder for missing files

    return render_template('upload_media.html', matches=matches)


@app.route('/get_progress')
def get_progress():
    return jsonify(progress=processing_progress)


@app.route('/search_incidents')
def search_incidents():
    # Get the media_path (or file name) from the request
    media_path = request.args.get('media_path')
    
    if media_path.startswith('/'):
        media_path = media_path[8:]
        print(media_path)

    if not media_path:
        return render_template('no_match.html', message="No media path provided.")
    
    # Search the Incident table for matching media_path
    incident = Incident.query.filter(Incident.media_path.ilike(f"%{media_path}%")).first()
    
    if incident:
        return render_template('incident_details.html', incident=incident)
    else:
        return render_template('no_match.html', message="No incidents found matching the media.")
    
@app.route('/write_message', methods=['POST'])
def write_message():
    data = request.get_json()
    
    sender_id = data.get('sender_id')
    receiver_id = data.get('receiver_id')
    message_text = data.get('message')
    
    if not sender_id or not receiver_id or not message_text:
        return jsonify({'error': 'Missing required fields'}), 400

    # Create a new message
    new_message = MessageInbox(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message=message_text,
        timestamp=datetime.utcnow(),
        status='sent',
        message_type='text',
        is_read=False
    )

    # Save to database
    db.session.add(new_message)
    db.session.commit()

    return jsonify({'message': 'Message sent successfully!', 'message_id': new_message.id}), 200

@app.route('/search_person_of_interest')
def search_person_of_interest():
    # Get the photo_path from the query parameter
    photo_path = request.args.get('photo_path')

    # Remove the first slash from the photo_path, if it exists
    if photo_path.startswith('/'):
        photo_path = photo_path[1:]
        photo_slashed = photo_path[8:]

    if photo_path:
        # Query the PersonOfInterest table based on the photo_path
        person = PersonOfInterest.query.filter_by(photo_path=photo_path).first()

        if person:
            # If a match is found, render a template with person details
            return render_template('person_details.html', person=person, photo_path=photo_slashed)
        else:
            # If no match is found, show a message
            return render_template('no_match.html', photo_path=photo_path, person=person)
    else:
        return "No photo path provided", 400
    
from flask import flash, redirect, url_for

# API route to submit incident
@app.route('/report', methods=['POST'])
def report():
    try:
        # Extract form data
        report_text = request.form.get('report_text')
        latitude = request.form.get('latitude')
        user_id = request.form.get('user_id')
        longitude = request.form.get('longitude')
        damage_estimate = request.form.get('damage_estimate')
        crops_affected = request.form.get('crops_affected')
        recommendation = request.form.get('recommendation')
        field_notes = request.form.get('field_notes')

        print("Form Data:", request.form)
        print("Files Data:", request.files)

        if not report_text or not latitude or not longitude:
            flash("Report text, latitude, and longitude are required", "error")
            return redirect(url_for('home'))

        latitude = float(latitude)
        longitude = float(longitude)

        # Handle optional media upload
        import uuid
        from werkzeug.utils import secure_filename


        # Handle media upload (image/video or captured video)
        media_path = None
        print("Files Data:", request.files)

        if 'media' in request.files:
            media_files = request.files.getlist('media')  # Get all uploaded files (list of files)

            for media_file in media_files:
        # Skip empty files
                if media_file.filename == '':
                    print("Skipping empty file")
                    continue
        
        # Check if the file type is allowed
                if allowed_file(media_file.filename):  
                    original_filename = secure_filename(media_file.filename)
                    base, ext = os.path.splitext(original_filename)
                    unique_filename = f"{base}_{uuid.uuid4().hex}{ext}"  # Create a unique filename

                    upload_folder = app.config['UPLOAD_FOLDER']
                    os.makedirs(upload_folder, exist_ok=True)

                    media_path = os.path.join(upload_folder, unique_filename)
                    media_file.save(media_path)  # Save the file

                    print(f"Media file saved to: {media_path}")
                else:
                    print(f"File type not allowed for file: {media_file.filename}")

# If no valid media was uploaded
        if not media_path:
            print("No valid media file processed")

        

            


        # Categorize the report
        #category = "Incident"
        category = categorize_incident(report_text)
        print("Incident Category:", category)  # Moved after category is assigned
        tokens = ", ".join([w for w in word_tokenize(report_text) if w.lower() not in stopwords.words('english')])
        openai_analysis = "Analysis pending."
        #openai_analysis = analyze_report(report_text)
        location = get_location(latitude, longitude)
        language = detect(report_text)

        # Create and commit incident
        new_incident = Incident(
            report_text=report_text,
            media_path=media_path,
            latitude=latitude,
            longitude=longitude,
            category=category,
            tokens=tokens,
            openai_analysis=openai_analysis,
            location=location,
            user_id=user_id,
            language=language,
            damage_estimate=damage_estimate,
            crops_affected=crops_affected,
            recommendation=recommendation,
            field_notes=field_notes,
            type="citizen-online"
        )

        print("New Incident:", vars(new_incident))  # Debug: Check the incident object
        
        db.session.add(new_incident)
        db.session.commit()

        print("Incident saved with ID:", new_incident.id)  # Check if ID was generated

        flash("Thank you for submitting your report! We appreciate your contribution.", "success")
        return redirect(url_for('home'))

    except ValueError:
        flash("Invalid latitude or longitude", "error")
        return redirect(url_for('home'))
    except Exception as e:
        db.session.rollback()
        print(f"Database error: {e}")
        flash(f"An error occurred: {e}", "error")
        return redirect(url_for('home'))

# Dashboard to view analysis

from math import ceil

from sqlalchemy import func

@app.route('/api/recent_incidents', methods=['GET'])
def recent_incidents():
    try:
        # Query the recent 50 incidents, excluding 'blotter'
        recent_incidents = Incident.query \
            .filter(Incident.type != 'blotter') \
            .order_by(Incident.timestamp.desc()) \
            .limit(50) \
            .all()
        
        # Convert to JSON format using the to_dict() method
        incident_data = [
            {
                "id": incident.id,
                "report_text": incident.report_text,
                "location": incident.location,
                "timestamp": incident.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            for incident in recent_incidents
        ]
        
        return jsonify(incident_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))

        if session['role'] != 'ADMIN':
            flash('You do not have the required permissions to access this page.', 'danger')
            return redirect(url_for('login'))

        # Get the start and end date from the query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        page = request.args.get('page', 1, type=int)  # Get the page number, default is 1
        per_page = request.args.get('per_page', 10, type=int)  # Get records per page, default is 10
        
        # Initialize filters
        start_date = None
        end_date = None
        
        # Convert the string date values to datetime objects if they are provided
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M')
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M')

        # Query incidents from the database with date filtering if dates are provided
        query = Incident.query
        
        if start_date and end_date:
            query = query.filter(Incident.timestamp >= start_date, Incident.timestamp <= end_date)
        elif start_date:
            query = query.filter(Incident.timestamp >= start_date)
        elif end_date:
            query = query.filter(Incident.timestamp <= end_date)

        # Exclude 'blotter' type records
        query = query.filter(Incident.type != 'blotter')

        # Apply pagination: .paginate(page, per_page, error_out=False) returns a Pagination object
        pagination = query.order_by(Incident.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
        
        incidents = pagination.items
        total_incidents = pagination.total  # Get the total number of incidents for pagination
        
        if not incidents:
            return jsonify({"message": "No data available"}), 404

        # Prepare the data to be displayed in the dashboard using the to_dict() method
        dashboard_data = [incident.to_dict() for incident in incidents]

        # Get the predictive analysis data (this could be a call to a function or API)
        recent_reports = Incident.query.order_by(Incident.timestamp.desc()).limit(50).all()
        prediction = "Increased risk of vandalism in urban areas at night."  # Example prediction

        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)

        # Query for incidents today vs yesterday
        incidents_today = Incident.query.filter(func.date(Incident.timestamp) == today).count()
        incidents_yesterday = Incident.query.filter(func.date(Incident.timestamp) == yesterday).count()

        # Query for responses today
        responses_today = ResponseDB.query.filter(func.date(ResponseDB.timestamp) == today).count()

        # Query for the top category today
        top_category_today = db.session.query(Incident.category, func.count(Incident.id).label('count'))\
            .filter(func.date(Incident.timestamp) == today)\
            .group_by(Incident.category)\
            .order_by(func.count(Incident.id).desc())\
            .first()

        if top_category_today:
            top_category_today = top_category_today.category
        else:
            top_category_today = 'None'

        # Calculate the start and end pages to limit the number of page links displayed
        start_page = max(pagination.page - 2, 1)
        end_page = min(pagination.page + 2, pagination.pages)

        # Media path (if you need it for any specific use)
        media_path = Incident.media_path

        # Pass the data to the template, including pagination data
        return render_template(
            'dashboard.html',
            username=session['username'],
            incidents=dashboard_data,
            prediction=prediction,
            media_path=media_path,
            incidents_today=incidents_today,
            incidents_yesterday=incidents_yesterday,
            responses_today=responses_today,
            top_category_today=top_category_today,
            pagination=pagination,  # Ensure pagination is passed to the template
            total_incidents=total_incidents,
            per_page=per_page,
            start_page=start_page,
            end_page=end_page
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#AUTOMATED ALERT

from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Message, Mail
import smtplib
import requests
from celery import Celery

celery = Celery(app.name, broker='redis://localhost:6379/0')  # configure your Celery broker

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Example SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'jasoncdelarosa@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'
mail = Mail(app)

# SMS Configuration (if using Twilio, for example)
TWILIO_PHONE_NUMBER = '+1234567890'  # Twilio phone number
TWILIO_SID = 'your_twilio_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'

@app.route('/set-alerts', methods=['GET', 'POST'])
def set_alerts():
    if request.method == 'POST':
        # Create a new alert from the form input
        name = request.form['name']
        urgency = request.form['urgency']
        location = request.form['location']
        category = request.form['category']
        alert_method = request.form['alert_method']
        contact_details = request.form['contact_details']
        
        new_alert = Alert(
            name=name,
            urgency=urgency,
            location=location,
            category=category,
            alert_method=alert_method,
            contact_details=contact_details
        )

        db.session.add(new_alert)
        db.session.commit()

        return redirect(url_for('view_alerts'))
    
    return render_template('set_alerts.html')

@app.route('/view-alerts', methods=['GET'])
def view_alerts():
    alerts = Alert.query.all()  # Fetch all alerts from the database
    return render_template('view_alerts.html', alerts=alerts)

@app.route('/edit-alert/<int:alert_id>', methods=['GET', 'POST'])
def edit_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)

    if request.method == 'POST':
        # Update the alert based on the form input
        alert.name = request.form['name']
        alert.urgency = request.form['urgency']
        alert.location = request.form['location']
        alert.category = request.form['category']
        alert.alert_method = request.form['alert_method']
        alert.contact_details = request.form['contact_details']
        
        db.session.commit()

        return redirect(url_for('view_alerts'))
    
    return render_template('edit_alert.html', alert=alert)

@app.route('/toggle-alert/<int:alert_id>', methods=['POST'])
def toggle_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    alert.is_active = not alert.is_active  # Toggle active/inactive status
    db.session.commit()
    return redirect(url_for('view_alerts'))

# Haversine formula to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
# Function to find hotspots
from collections import defaultdict
from geopy.distance import geodesic

def find_hotspots(incidents, radius_km):
    """
    Group incidents into hotspots within a given radius (in km).
    """
    hotspots = defaultdict(lambda: {"count": 0, "category": set()})

    for i, incident in enumerate(incidents):
        if incident.latitude and incident.longitude:
            location = (incident.latitude, incident.longitude)

            # Check if it falls within an existing hotspot
            found = False
            for key, value in hotspots.items():
                hotspot_location = key
                distance = geodesic(location, hotspot_location).km
                if distance <= radius_km:
                    value["count"] += 1
                    value["category"].add(incident.category)
                    found = True
                    break

            # If not close to an existing hotspot, create a new one
            if not found:
                hotspots[(incident.latitude, incident.longitude)]["count"] = 1
                hotspots[(incident.latitude, incident.longitude)]["category"] = {incident.category}

    # Convert to a list of dictionaries for JSON
    hotspot_list = [
        {
            "latitude": lat,
            "longitude": lon,
            "count": data["count"],
            "category": ", ".join(data["category"]),
        }
        for (lat, lon), data in hotspots.items()
    ]

    return hotspot_list

@app.route('/hotspots')
def hotspots():
    incidents = db.session.query(Incident).filter(
        Incident.latitude.isnot(None),
        Incident.longitude.isnot(None)
    ).all()
    radius = 1.0  # Radius in kilometers
    hotspots = find_hotspots(incidents, radius)
    return render_template('hotspots.html',hotspots=json.dumps(hotspots))
    
@app.route('/get_markers_map')
def get_markers_map():
    markers = Marker.query.all()
    markers_data = [{
        'id': marker.id,
        'label': marker.label,
        'description': marker.description,
        'latitude': marker.latitude,
        'longitude': marker.longitude,
        'category': marker.category,
        'created_at': marker.created_at.isoformat()
    } for marker in markers]
    return jsonify(markers_data)
# Generate sample data
@app.route('/generate_sample_data', methods=['POST'])
def generate_sample_data():
    sample_texts = [
        "Robbery occurred in the downtown area.",
        "A car was vandalized last night.",
        "Victim reported harassment in the workplace.",
        "Suspicious activity reported near the mall.",
        "A fraudulent check was cashed at the local bank.",
        "An assault was reported near the subway station."
    ]
    for _ in range(100):
        sample_text = random.choice(sample_texts)
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        timestamp = datetime.utcnow() - timedelta(days=random.randint(0, 365))

        incident = Incident(
            report_text=sample_text,
            media_path=None,
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,  # Ensure this is a datetime object
            category=categorize_incident(sample_text),
            tokens=" ".join([w for w in word_tokenize(sample_text) if w.lower() not in stopwords.words('english')]),
            openai_analysis=analyze_report(sample_text),
            location=f"Latitude: {latitude}, Longitude: {longitude}",
        )
        db.session.add(incident)
    db.session.commit()
    return jsonify({"message": "100 sample incidents generated"}), 201

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
