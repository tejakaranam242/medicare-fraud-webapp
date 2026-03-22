-- Create DB
CREATE DATABASE IF NOT EXISTS fraud_detection;
USE fraud_detection;

-- Users table (for registration & login)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table (for storing logs)
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    user_input JSON NOT NULL,
    model VARCHAR(50) NOT NULL,
    prediction INT NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
