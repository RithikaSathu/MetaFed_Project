# MetaFed Project

## Overview
MetaFed is a federated learning-based system designed to train personalized models across distributed clients while preserving data privacy.

The system integrates multiple federated algorithms such as:
- FedAvg
- FedProx
- FedBN
- MetaFed (proposed approach)

## Features
- Federated training across multiple clients
- Personalized model adaptation
- Comparison of different FL algorithms
- Visualization of training performance
- Frontend dashboard for interaction

## Tech Stack
- Frontend: React + TypeScript + Tailwind CSS
- Backend: Python (Flask/FastAPI)
- ML: PyTorch / Custom models
- Tools: Vite

## Project Structure
backend/ → APIs + training logic  
frontend/ → UI dashboard  
metafed_training/ → core FL algorithms  

## Setup Instructions

### Backend
cd backend
pip install -r requirements.txt
python app.py

### Frontend
cd frontend
npm install
npm run dev
