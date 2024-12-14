from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar, QGridLayout, QGraphicsDropShadowEffect, QSizePolicy,QTableWidget,QTableWidgetItem
)
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QSlider
import csv
import re
import time
import os

import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FixedLocator


import nltk
from nltk.corpus import stopwords

# Ensure you have the stopwords downloaded
nltk.download('stopwords')

class ModernUI(QWidget):
    def __init__(self):
        super().__init__()

        QApplication.setFont(QFont("Times New Roman", 12))

        # Initialize the UI components and layouts
        self.setWindowTitle("Extractive Summarization Tool")
        self.setGeometry(100, 100, 1500, 980)
        self.set_custom_palette()

        # Main layout
        main_layout = QVBoxLayout()

        # Create a main container for header, body, and footer
        self.main_container = QWidget()
        self.main_container_layout = QVBoxLayout(self.main_container)
        self.main_container.setStyleSheet("""
            background-color: white;
            border-radius: 15px;
            padding: 5px; 
            margin: 20px; 
        """)
        self.main_container.setFixedHeight(int(980 * 0.80))

        # Create title labels
        title_label1 = QLabel("-ExtractSum-")
        title_label = QLabel("An Extractive Summarization of Terms of Service Using Non-Negative Matrix Factorization with Global Vectors and Rhetorical Sentence Features")
        title_label.setFont(QFont("Times New Roman", 26))
        title_label1.setFont(QFont("Arial", 36))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("margin-left: 130px;margin-right: 130px;")  
        title_label1.setAlignment(Qt.AlignCenter)
        title_label1.setWordWrap(True)
        title_label1.setStyleSheet("margin-left: 130px;margin-right: 130px;")

        # Add navigation buttons
        self.next_button = QPushButton(">")
        self.prev_button = QPushButton("<")
        self.prev_button.hide()  # Initially hide the "<" button

        # Style and set fixed size for the buttons
        self.next_button.setFixedSize(30, 30)
        self.prev_button.setFixedSize(30, 30)
        self.next_button.setStyleSheet("color: black;")
        self.prev_button.setStyleSheet("color: black;")

        # Create a button layout for placing the button on the top left above the title
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.prev_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.next_button, alignment=Qt.AlignLeft)

        # Add button layout before the title labels
        main_layout.addLayout(button_layout)

        main_layout.addWidget(title_label1)
        main_layout.addWidget(title_label)

        # Call functions to create header, body, and footer of the main container
        self.create_header(self.main_container_layout)
        self.create_body(self.main_container_layout)
        self.create_footer(self.main_container_layout)

        # Create the hidden container
        self.hidden_container = QWidget()
        self.hidden_container_layout = QVBoxLayout(self.hidden_container)
        
        # Set the same stylesheet for the hidden container as the main container
        self.hidden_container.setStyleSheet("""
            background-color: white;
            border-radius: 15px;
            padding: 5px; 
            margin: 20px; 
        """)
        self.hidden_container.setFixedHeight(int(980 * 0.80))  # Match the height of the main container



        # Set the hidden container's layout
        self.hidden_container.setLayout(self.hidden_container_layout)
        self.hidden_container.hide()  # Initially hide the hidden container
        
        
        self.create_header2(self.hidden_container_layout)
        self.create_header3(self.hidden_container_layout)
        self.create_body2(self.hidden_container_layout)

        # Add the main container and hidden container to the main layout
        main_layout.addWidget(self.main_container)
        main_layout.addWidget(self.hidden_container)

        self.setLayout(main_layout)

        # Connect the next button to show/hide containers
        self.next_button.clicked.connect(self.toggle_hidden_container)
        
        self.uploaded_files = {button: [] for button in self.file_counts.keys()}

        self.glove_vectors = None  # To store GloVe vectors

        self.keywords = {
            "Obligations": ["needs", "required", "shall", "must", "compliance", "bound", "obligated", "agrees", "agreed", "agree", "committed", "commit", "duty", "responsible"],
            "Rights": ["can", "may", "reserves", "right to", "right at", "permitted", "authority", "authorize", "entitled", "privilege"],
            "Conditions": ["if", "provided", "subject to", "case", "unless", "otherwise", "under", "condition"],
            "Exclusions of Liability": ["responsibility", "liable", "free", "liability", "excludes", "accountable", "disclaims"],
            "Warranties and Disclaimers": ["warranty", "warranties", "guarantee", "guaranteed", "guarantees", "assurance"],
            "Contrast/Concession": ["products and services", "but", "however", "even", "although", "notwithstanding", "despite", "spite", "nevertheless", "nonetheless", "contrast"],
            "Cause and Effect": ["access to", "because", "result", "thus", "consequently", "since", "therefore", "consequence", "hence"],
            "Addition": ["terms and conditions", "products or services", "also", "additionally", "as well as", "furthermore", "moreover", "besides"],
            "Comparison/Similarity": ["way", "similar", "same", "similarly", "equally"],
            "Condition": ["long as"],
            "Purpose": ["in order", "time to time", "information that", "so that", "purpose", "goal", "objective", "intention", "aim"]
        }

    def toggle_hidden_container(self):
        """Toggle the visibility of the hidden container."""
        if self.hidden_container.isVisible():
            self.hidden_container.hide()
            self.main_container.show()
        else:
            self.main_container.hide()
            self.hidden_container.show()

    
    def write_results_to_csv(self, results):
        with open('rouge_results.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Summary Type', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
            for result in results:
                # Assuming the result string is formatted as 'key:  R1=value R2=value RL=value'
                parts = result.split(' ')
                key = parts[0].strip(':')
                r1 = parts[1].split('=')[1]
                r2 = parts[2].split('=')[1]
                rl = parts[3].split('=')[1] if len(parts) > 3 else '0'
                writer.writerow([key, r1, r2, rl])

    def clear_uploaded_files(self):
        # Clear all uploaded files and reset counts
        self.uploaded_files.clear()
        
        # Reinitialize the uploaded_files dictionary with empty lists for each button
        self.uploaded_files = {button: [] for button in ["REF", "1 NMF", "2 NMFG", "3 NMFGR", "4 NMFGS", "5 NMFGC",
                                                    "6 NMFGRSC", "7 NMFGRC", "8 NMFGRS",  "9 NMFRSC", "A NMFGRSC"]}
        
        # Reset the file counts to zero
        self.file_counts = {button: 0 for button in self.file_counts}
        
        # Clear the results display
        self.results_display.clear()
        
        print("All uploaded files have been cleared.")
        self.results_display.append("All uploaded files have been cleared.")

    def upload_files(self, button_name):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, f"Select files for {button_name}", "",
                                                "Text Files (*.txt);;All Files (*)", options=options)
        try:
            if files:
                # Update file count for the button
                self.file_counts[button_name] += len(files)
                # Store uploaded files
                self.uploaded_files[button_name].extend(files)
                print(f"Uploaded {len(files)} files for {button_name}. Total: {self.file_counts[button_name]}")
                self.results_display.append(f"Uploaded {len(files)} files for {button_name}. Total: {self.file_counts[button_name]}")
            else:
                raise ValueError("No files were uploaded.")
        except Exception as e:
            error_message = f"Error uploading files for {button_name}: {str(e)}"
            print(error_message)  # Debug output to console
            self.results_display.append(error_message)  # Show error in the results display

    def get_uploaded_files(self, button_name):
        return self.uploaded_files.get(button_name, [])  # Return the list of uploaded files for the button
    
    

    def create_body2(self, main_layout):
        body_layout = QVBoxLayout()  # Use QVBoxLayout to stack results

        # Create a container for the text areas
        container_widget = QWidget()
        container_widget.setStyleSheet("""
            margin-top:0px;
            padding:15px;
            color:black;
            font-family: 'Arial'; 
            font-size: 12pt;
            border:  1px solid  #D3D3D3; 
        """)
        
        # Create a QTextEdit to display results (for errors, logs, etc.)
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)  # Make it read-only
        body_layout.addWidget(self.results_display)
        # Create a QTableWidget for CSV table display (initially hidden)
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)  # Example: 4 columns for ROUGE-1, ROUGE-2, ROUGE-L, Key
        self.table_widget.setHorizontalHeaderLabels(['Key', 'Average ROUGE-1', 'Average ROUGE-2', 'Average ROUGE-L'])
        self.table_widget.setVisible(False)  # Hide it by default
        self.table_widget.setFixedHeight(450)

        # Set the style for the table widget
        self.table_widget.setStyleSheet("""
            QTableWidget {
                margin-top: 30px;
                padding: 5px;
                color: black;
                font-family: 'Arial'; 
                font-size: 12pt;
                border: 1px solid #D3D3D3; 
                gridline-color: #D3D3D3;
            }
            QTableWidget::item {
                padding: 5px;  /* Add padding for better spacing */
                text-align: AlignCenter; /* Center align text in cells */
            }
            QHeaderView::section {
                background-color: #f0f0f0; /* Light gray background for headers */
                color: black;  /* Black text color for headers */
                font-weight: bold;  /* Bold text for headers */
                border: 1px solid #D3D3D3; /* Border around headers */
            }
        """)

        # Set specific widths for the columns
        self.table_widget.setColumnWidth(0, 250)  # Width for 'Key' column
        self.table_widget.setColumnWidth(1, 250)  # Width for 'Average ROUGE-1' column
        self.table_widget.setColumnWidth(2, 250)  # Width for 'Average ROUGE-2' column
        self.table_widget.setColumnWidth(3, 250)  # Width for 'Average ROUGE-L' column
        
                # Ensure the header is visible
        self.table_widget.horizontalHeader().setVisible(True)

        # Optional: Set alignment for all cells
        for i in range(self.table_widget.rowCount()):
            for j in range(self.table_widget.columnCount()):
                item = self.table_widget.item(i, j)
                if item is not None:  # Check if item exists
                    item.setTextAlignment(Qt.AlignCenter)  # Center align text

        # Add the table widget to your layout as needed
        body_layout.addWidget(self.table_widget)

        # Set the layout to the container widget
        container_widget.setLayout(body_layout)
        container_widget.setFixedHeight(500)  

        # Add the container widget to the main layout
        main_layout.addWidget(container_widget)

    def create_header2(self, main_layout):
        self.active_button = None
        self.file_counts = {button: 0 for button in ["REF", "1 NMF", "2 NMFG", "3 NMFGR", "4 NMFGS", "5 NMFGC",
                                                    "6 NMFGRSC", "7 NMFGRC", "8 NMFGRS",  "9 NMFRSC", "A NMFGRSC"]}
        
        header_layout = QVBoxLayout()

        container_widget = QWidget()
        container_layout = QHBoxLayout()
        container_widget.setStyleSheet("""
            background-color: white;
            border-top: none;
            border-left: none;
            border-right: none;
            border-bottom: 1px solid  #D3D3D3;
            border-radius: none;
        """)
        container_widget.setFixedHeight(int(100))
        container_widget.setFixedWidth(self.width())

        buttons = ["REF", "1 NMF", "2 NMFG", "3 NMFGR", "4 NMFGS", "5 NMFGC", "6 NMFGRSC", "7 NMFGRC", "8 NMFGRS",  "9 NMFRSC", "A NMFGRSC"]

        default_style = """
            QPushButton {
                background-color: white;
                color: black;
                font-family: Arial;
                font-size: 12pt;
                padding: 0px;
                margin-bottom: 15px;
                border: none;
            }
        """

        active_style = """
            QPushButton {
                background-color: white;
                color: green;
                font-family: Arial;
                font-size: 12pt;
                padding: 3px;
                margin-bottom: 15px;
                border: none;
                border-bottom: 2px solid green;
                border-radius: 0px;
                font-weight: bold;
            }
        """

        for button_name in buttons:
            button = QPushButton(button_name)
            button.setStyleSheet(default_style)
            button.setFixedHeight(int(80))
            container_layout.addWidget(button)

            # Connect button click to file upload dialog
            button.clicked.connect(lambda checked, btn=button_name: self.upload_files(btn))

            # Connect button click to style change
            button.clicked.connect(lambda checked, btn=button: self.update_active_button(btn, default_style, active_style))

        container_widget.setLayout(container_layout)
        header_layout.addWidget(container_widget)

        # Add header layout to the main layout
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        main_layout.addWidget(header_widget)

    def create_header3(self, main_layout):
        self.active_button = None

        header_layout = QVBoxLayout()

        container_widget = QWidget()
        container_layout = QHBoxLayout()
        container_widget.setStyleSheet("""
            background-color: white;
            border: 1px solid  #D3D3D3;
            border-radius: none;
        """)
        container_widget.setFixedHeight(int(100))
        container_widget.setFixedWidth(self.width())

        buttons = ["Clear Upload","Calculate"]

        default_style = """
            QPushButton {
                background-color: white;
                color: black;
                font-family: Arial;
                font-size: 12pt;
                padding: 0px;
                margin-bottom: 15px;
                border: none;
            }
        """

        active_style = """
            QPushButton {
                background-color: white;
                color: green;
                font-family: Arial;
                font-size: 12pt;
                padding: 3px;
                margin-bottom: 15px;
                border: none;
                border-bottom: 2px solid green;
                border-radius: 0px;
                font-weight: bold;
            }
        """

        for button_name in buttons:
            button = QPushButton(button_name)
            button.setStyleSheet(default_style)
            button.setFixedHeight(int(80))
            container_layout.addWidget(button)

            # Connect button click to style change
            button.clicked.connect(lambda checked, btn=button: self.update_active_button(btn, default_style, active_style))
            
            # Connect the "Calculate" button to the calculation method
            if button_name == "Calculate":
                button.clicked.connect(self.calculate_rouge_scores)
            # Connect the "Clear Upload" button to the clear method
            if button_name == "Clear Upload":
                button.clicked.connect(self.clear_uploaded_files)  # Add this line

        container_widget.setLayout(container_layout)
        header_layout.addWidget(container_widget)
        
        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setMinimum(0)
        self.progress_bar2.setMaximum(100)
        container_layout.addWidget(self.progress_bar2)  # Add progress bar to the layout


        # Add header layout to the main layout
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        main_layout.addWidget(header_widget)
        
    def compute_rouge(self, ref_file, gen_file):
        # You might want to read the contents of the files first
        try:
            with open(ref_file, 'r', encoding='utf-8') as f:
                reference_text = f.read()
        except Exception as e:
            print(f"Error reading reference file {ref_file}: {e}")
            return 0, 0, 0  # or some appropriate error handling

        try:
            with open(gen_file, 'r', encoding='utf-8') as f:
                generated_text = f.read()
        except Exception as e:
            print(f"Error reading generated file {gen_file}: {e}")
            return 0, 0, 0  # or some appropriate error handling

        # Here you would compute ROUGE scores. 
        # For example, if using the rouge_score library:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)

        # Return the scores in the format you need
        return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

    def calculate_rouge_scores(self):
        try:
            # Get the uploaded file names
            reference_files = self.get_uploaded_files('REF')
            if not reference_files:
                raise ValueError("No reference files uploaded.")

            # Proceed even if there are no common filenames, just work with available files
            all_filenames = {os.path.splitext(os.path.basename(f))[0]: f for f in reference_files}

            # Prepare for ROUGE calculation
            summary_folder = "Summary"
            rouge_score_folder = os.path.join(summary_folder, "ALL_ROUGE_SCORE")
            os.makedirs(rouge_score_folder, exist_ok=True)

            total_comparisons = len(reference_files)  # You can adjust this if you want more precise tracking
            self.progress_bar2.setValue(0)  # Reset the progress bar
            progress = 0

            # Initialize list to store average scores
            avg_scores = []

            # Iterate over non-REF buttons and calculate ROUGE for available files
            buttons = ["1 NMF", "2 NMFG", "3 NMFGR", "4 NMFGS", "5 NMFGC", "6 NMFGRSC", "7 NMFGRC", "8 NMFGRS", "9 NMFRSC", "A NMFGRSC"]
            for button in buttons:
                uploaded_files = self.get_uploaded_files(button)
                if not uploaded_files:
                    continue

                # Initialize score lists for this button
                button_scores_r1 = []
                button_scores_r2 = []
                button_scores_rl = []

                # Open a CSV for storing ROUGE scores
                csv_path = os.path.join(rouge_score_folder, f"{button}_rouge_scores.csv")
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Filename', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])

                    for ref_file in reference_files:
                        ref_filename = os.path.splitext(os.path.basename(ref_file))[0]

                        # Check if there is a matching generated file for this button
                        matching_gen_files = [gen_file for gen_file in uploaded_files if os.path.splitext(os.path.basename(gen_file))[0] == ref_filename]
                        
                        if not matching_gen_files:
                            # No matching file found for this reference file, skip it
                            continue

                        # There is a matching generated file, calculate ROUGE
                        gen_file = matching_gen_files[0]  # Take the first matching file
                        R1, R2, RL = self.compute_rouge(ref_file, gen_file)
                        button_scores_r1.append(R1)
                        button_scores_r2.append(R2)
                        button_scores_rl.append(RL)

                        # Write file-level ROUGE scores to the CSV
                        writer.writerow([ref_filename, R1, R2, RL])

                        # Update progress bar
                        progress += 1
                        progress_percentage = int((progress / total_comparisons) * 100)
                        self.progress_bar2.setValue(progress_percentage)

                # Calculate average ROUGE scores for this button
                if button_scores_r1:
                    mean_r1 = np.mean(button_scores_r1)
                    mean_r2 = np.mean(button_scores_r2)
                    mean_rl = np.mean(button_scores_rl)
                    avg_scores.append([button, mean_r1, mean_r2, mean_rl])

            # Write overall average CSV
            avg_csv_path = os.path.join(rouge_score_folder, "average_rouge_scores.csv")
            with open(avg_csv_path, 'w', newline='') as avg_csvfile:
                avg_writer = csv.writer(avg_csvfile)
                avg_writer.writerow(['Key', 'Average ROUGE-1', 'Average ROUGE-2', 'Average ROUGE-L'])
                avg_writer.writerows(avg_scores)

            # Populate the results table with average scores
            self.results_display.setVisible(False)  # Hide text display
            self.table_widget.setVisible(True)  # Show table
            self.table_widget.setRowCount(0)
            for i, row in enumerate(avg_scores):
                self.table_widget.insertRow(i)
                for j, value in enumerate(row):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(value)))

            # Plot the scores if needed
            self.plot_average_scores(avg_scores)

        except Exception as e:
            error_message = f"Error calculating ROUGE scores: {str(e)}"
            print(error_message)
            self.results_display.append(error_message)
            self.table_widget.setVisible(False)
            self.results_display.setVisible(True)

    def plot_average_scores(self, avg_scores):
        # Prepare data for plotting
        keys = [row[0] for row in avg_scores]
        avg_r1 = [row[1] for row in avg_scores]
        avg_r2 = [row[2] for row in avg_scores]
        avg_rl = [row[3] for row in avg_scores]

        # Create a new QWidget for the plots
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create the plots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # ROUGE-1 Plot
        axs[0].bar(keys, avg_r1, color='blue', alpha=0.7)
        axs[0].set_title('Average ROUGE-1 Scores')
        axs[0].set_ylabel('ROUGE-1 Score')
        axs[0].set_xticklabels(keys, rotation=45, ha='right')

        # ROUGE-2 Plot
        axs[1].bar(keys, avg_r2, color='green', alpha=0.7)
        axs[1].set_title('Average ROUGE-2 Scores')
        axs[1].set_ylabel('ROUGE-2 Score')
        axs[1].set_xticklabels(keys, rotation=45, ha='right')

        # ROUGE-L Plot
        axs[2].bar(keys, avg_rl, color='red', alpha=0.7)
        axs[2].set_title('Average ROUGE-L Scores')
        axs[2].set_ylabel('ROUGE-L Score')
        axs[2].set_xticklabels(keys, rotation=45, ha='right')

        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()  # Or embed this in your PyQt widget as needed

        # Add the plot widget to your main window layout
        self.layout().addWidget(plot_widget)
    
    def find_common_filenames(self):
        # Collect file names from all buttons except 'REF'
        buttons = ["NMF", "NMFG", "NMFGR", "NMFGS", "NMFGC", "NMFGRS", 
                "NMFGRC", "NMFGRSC", "NMFGRSCG", "NMFGRSC"]

        # Get the sets of filenames for each button (only base names without extensions)
        file_sets = []
        for button in buttons:
            uploaded_files = self.get_uploaded_files(button)
            if uploaded_files:  # If there are files uploaded for this button
                filenames = {os.path.splitext(os.path.basename(f))[0] for f in uploaded_files}
                file_sets.append(filenames)

        # If no files were uploaded, return an empty set
        if not file_sets:
            return set()

        # Find the intersection of all file sets (common filenames across all buttons)
        common_filenames = set.intersection(*file_sets) if file_sets else set()
        return common_filenames



    

    def set_custom_palette(self):
        dark_palette = QPalette()
        background_color = QColor("#1E1E1E")  
        button_background = QColor("#1B1A55")
        dark_palette.setColor(QPalette.Window, background_color)
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.Button, button_background)
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        QApplication.setPalette(dark_palette)       
    
    def create_body(self, main_layout):
        body_layout = QHBoxLayout()
        
        # Create a container for the text areas
        container_widget = QWidget()
        container_widget.setStyleSheet("""
            margin-top:0px;
            padding:0px;
            color:black;
            font-family: 'Arial'; 
            font-size: 12pt; 
        """)

        # Create a vertical layout for the left text area and its title
        left_layout = QVBoxLayout()
        
        # Title for the left text area
        left_title = QLabel("Upload Text File:")
        left_title.setAlignment(Qt.AlignCenter)  
        left_layout.addWidget(left_title) 
        left_title.setStyleSheet("font-weight: bold;") 
        
        # Left text area
        self.left_text_area = QTextEdit()
        self.left_text_area.setReadOnly(True)
        self.left_text_area.setStyleSheet("""
            background-color: white;
            color: black;
            border: 2px solid #D3D3D3;  
            border-radius: 15px;
            font-family: 'Arial'; 
            font-size: 12pt;  
            padding: 10px; 
            margin-right:-0.5px; 
        
        """)
        left_layout.addWidget(self.left_text_area) 

        
        right_layout = QVBoxLayout()
        
        # Title for the right text area
        right_title = QLabel("Results:")
        right_title.setAlignment(Qt.AlignCenter) 
        right_layout.addWidget(right_title)  
        right_title.setStyleSheet("font-weight: bold;padding:0px") 
        
        # Right text area
        self.right_placeholder = QTextEdit()
        self.right_placeholder.setReadOnly(True)
        self.right_placeholder.setStyleSheet("""
            background-color: white;
            color: black;
            border: 2px solid #D3D3D3;  
            border-radius: 15px;
            font-family: 'Arial'; 
            font-size: 12pt;  
            padding: 10px;  
            margin-left:-0.5px; 
        """)
        right_layout.addWidget(self.right_placeholder) 

        
        body_layout.addLayout(left_layout)
        body_layout.addLayout(right_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Set the layout to the container widget
        container_widget.setLayout(body_layout)
        container_widget.setFixedHeight(int(550))  
        
        # Add the container widget and the progress bar to the main layout
        main_layout.addWidget(container_widget)
        main_layout.addWidget(self.progress_bar)

    def create_footer(self, main_layout):
        footer_layout = QHBoxLayout()

        
        container_widget = QWidget()
        container_layout = QHBoxLayout(container_widget)
        container_widget.setStyleSheet("""
            background-color: white;
            color: black;
            margin-top: 0px;
            border-top: 1px solid  #D3D3D3; 
            border-left: none;           
            border-right: none;         
            border-bottom: none;        
            border-radius: 0px;         
        """)

        container_widget.setFixedHeight(int(80))
    

        # Upload buttons
        left_side_layout = QHBoxLayout()
        self.file_input_button = QPushButton("Upload TXT File")
        self.file_input_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-family: Arial;
                font-size: 12pt;
                border: none;
                margin-top:0px;
                padding:0px;
                font-weight: bold; 
            }
        """)
        self.file_input_button.setFixedSize(180, 50)
        self.file_input_button.clicked.connect(self.open_file_dialog)
        
        self.batch_upload_button = QPushButton("Batch Upload TXT Files")
        self.batch_upload_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-family: Arial;
                font-size: 12pt;
                border: none;
                margin-top:0px;
                padding:0px;
                font-weight: bold; 
            }
        """)
        self.batch_upload_button.setFixedSize(220, 50)
        self.batch_upload_button.clicked.connect(self.open_batch_file_dialog)
        
        left_side_layout.addWidget(self.file_input_button)
        left_side_layout.addWidget(self.batch_upload_button)

        # Sentence count, font size slider, and summarize buttons
        right_side_layout = QHBoxLayout()
        self.sentence_count_label = QLabel("Sentence Count: 0")
        self.sentence_count_label.setAlignment(Qt.AlignCenter)
        self.sentence_count_label.setStyleSheet("border: none;")  

        # Font Size Label
        self.font_size_label = QLabel("Font Size:")
        self.font_size_label.setAlignment(Qt.AlignCenter)
        self.font_size_label.setStyleSheet("border: none;") 
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setStyleSheet("border: none;")  # Remove border
        self.font_size_slider.setRange(10, 24)
        self.font_size_slider.setValue(12)
        self.font_size_slider.setFixedSize(150, 30)
        self.font_size_slider.valueChanged.connect(self.update_font_size)

        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.setStyleSheet("""
            QPushButton {
                background-color: #5C946E ;
                color: white;
                font-family: Arial;
                font-size: 12pt;
                border: none;
                margin-top:0px;
                padding:0px;
                font-weight: bold; 
                border-radius:15px;
            }
        """)
        self.summarize_button.setFixedSize(180, 70)
        self.summarize_button.clicked.connect(self.summarize_document)

        self.batch_summarize_button = QPushButton("Batch Summarize")
        self.batch_summarize_button.setStyleSheet("""
            QPushButton {
                background-color: #5C946E ;
                color: white;
                font-family: Arial;
                font-size: 12pt;
                border: none;
                padding:0px;
                margin-top:0px;
                font-weight: bold; 
                border-radius:15px;
            }
        """)
        self.batch_summarize_button.setFixedSize(200, 70)
        self.batch_summarize_button.clicked.connect(self.batch_summarize_documents)
        
        progress_layout = QVBoxLayout()

        # Progress label
        self.progress_label = QLabel("Progress Status:")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 10pt;border:none;margin-top:0px;paddinng:0px")

        progress_layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)  

        
        self.progress_bar.setFixedHeight(40)  
        self.progress_bar.setStyleSheet("QProgressBar { min-width: 100px; }")  

        progress_layout.addWidget(self.progress_bar)

        
        right_side_layout.addLayout(progress_layout)


        
        right_side_layout.addWidget(self.sentence_count_label)
        right_side_layout.addWidget(self.font_size_label)
        right_side_layout.addWidget(self.font_size_slider)
        right_side_layout.addWidget(self.summarize_button)
        right_side_layout.addWidget(self.batch_summarize_button)

        
        container_layout.addLayout(left_side_layout)
        container_layout.addStretch(1)  
        container_layout.addLayout(right_side_layout)

        footer_layout.addWidget(container_widget)
        main_layout.addLayout(footer_layout)

    def create_header(self, main_layout):
        self.active_button = None  

        header_layout = QVBoxLayout()  

        
        container_widget = QWidget()
        container_layout = QHBoxLayout()
        container_widget.setStyleSheet("""
            background-color: white;
            border-top: none; 
            border-left: none;           
            border-right: none;         
            border-bottom: 1px solid  #D3D3D3; 
            border-radius:none;
            
        """)
        container_widget.setFixedHeight(int(100))
        container_widget.setFixedWidth(self.width())
        
        buttons = [
            "Pre-process", "Pre-defined Keywords", "NMF", 
            "NMFScore", "FeatureScore",
            "Surface Feature", "Content Feature", "Rhetorical Feature", 
            "Sentence Extraction"
        ]

       
        default_style = """
            QPushButton {
                background-color: white;
                color: black;
                font-family: Arial;
                font-size: 12pt;
                padding:0px;
                margin-bottom:15px;
                border: none;
                
            }
        """

    
        active_style = """
            QPushButton {
                background-color: white;
                color: green;
                font-family: Arial;
                font-size: 12pt;
                padding: 3px;
                margin-bottom: 15px;
                border: none;
                border-bottom: 2px solid green;
                border-radius:0px;
                font-weight: bold; 

            }
        """

        for button_name in buttons:
            button = QPushButton(button_name)
            button.setStyleSheet(default_style)
            button.setFixedHeight(int(80))
            container_layout.addWidget(button)


            if button_name == "Pre-process":
                button.clicked.connect(self.preprocess_file)
            elif button_name == "NMF":
                button.clicked.connect(self.perform_nmf)
            elif button_name == "NMFScore":
                button.clicked.connect(self.calculate_nmf_score)
            elif button_name == "Surface Feature":
                button.clicked.connect(self.calculate_surface_features)
            elif button_name == "Content Feature":
                button.clicked.connect(self.calculate_content_features)
            elif button_name == "Pre-defined Keywords":
                button.clicked.connect(self.analyze_keywords)
            elif button_name == "Rhetorical Feature":
                button.clicked.connect(self.calculate_rhetorical_features)
            elif button_name == "FeatureScore":
                button.clicked.connect(self.calculate_feature_score)
            elif button_name == "Sentence Extraction":
                button.clicked.connect(self.extract_sentences)

            # Connect button click to style change
            button.clicked.connect(lambda checked, btn=button: self.update_active_button(btn, default_style, active_style))

        container_widget.setLayout(container_layout)
        header_layout.addWidget(container_widget)

        # Add header layout to the main layout
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        main_layout.addWidget(header_widget)

    def update_active_button(self, button, default_style, active_style):
        # If there is an active button, reset its style
        if self.active_button:
            self.active_button.setStyleSheet(default_style)

        
        button.setStyleSheet(active_style)
        self.active_button = button
 
    def open_batch_file_dialog(self):
        self.right_placeholder.clear()
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Multiple TXT Files", "", "Text Files (*.txt)", options=options)
        
        if files:
            # Store the selected files in a list attribute
            self.file_list = files
            file_list_text = "\n".join([os.path.basename(file) for file in files])
            total_files = len(files)
            display_text = f"Total Files Uploaded: {total_files}\n\nFiles:\n{file_list_text}"
            self.left_text_area.setPlainText(display_text)
        else:
            self.left_text_area.setPlainText("No files selected.")
            self.file_list = []
    
    def open_file_dialog(self):
        self.right_placeholder.clear()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open TXT File", "", "Text Files (*.txt)", options=options)  # Allow only TXT files
        if file_name:
            self.file_name = file_name  # Store the selected file name
            print(f"File selected: {self.file_name}")  
            self.load_file_content(file_name)
        else:
            print("No file selected")  
    
    def load_file_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.left_text_area.setPlainText(content)
        except Exception as e:
            self.left_text_area.setPlainText(f"Error reading file: {e}")
    
    def update_font_size(self):
        font_size = self.font_size_slider.value()
        font = QFont("Arial", font_size)
        self.left_text_area.setFont(font)
        self.right_placeholder.setFont(font)
        
     
        left_style = f"""
            background-color: white;
            color: black;
            border: 2px solid #D3D3D3;  /* Light gray border */
            border-radius: 15px;  /* Rounded corners */
            font-family: 'Times New Roman';
            font-size: {font_size}pt;  /* Set dynamic font size */
            padding: 10px;  /* Add some padding inside the text area */
        """
        
        right_style = f"""
            background-color: white;
            color: black;
            border: 2px solid #D3D3D3;  /* Light gray border */
            border-radius: 15px;  /* Rounded corners */
            font-family: 'Times New Roman';
            font-size: {font_size}pt;  /* Set dynamic font size */
            padding: 10px;  /* Add some padding inside the text area */
        """

        self.left_text_area.setStyleSheet(left_style)
        self.right_placeholder.setStyleSheet(right_style)
    
    def load_glove_vectors(self, glove_file):
        glove_vectors = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=float)
                glove_vectors[word] = vector
        return glove_vectors    
    
    
    def preprocess_file(self):
        try:
            # Check if a file has been selected
            if not hasattr(self, 'file_name') or not self.file_name:
                self.right_placeholder.setPlainText("No file selected. Please upload a file first.")
                print("Error: No file selected in preprocess_file")
                return

            # Get the content from the left text area
            content = self.left_text_area.toPlainText()
            if not content:
                self.right_placeholder.setPlainText("No content to process. Please upload a file first.")
                print("Error: No content to process")
                return

            print(f"Preprocessing started for file: {self.file_name}")
            print(f"Content length: {len(content)} characters")

            # Ensure content is a string before applying string methods
            if not isinstance(content, str):
                content = str(content)

            # Convert content to lowercase (optional)
            content = content.lower()

            # Create Summary folder if not exists
            if not os.path.exists("Summary"):
                os.makedirs("Summary")
                print("Created 'Summary' folder")

            # Create a folder named after the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder for file: {folder_path}")

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Split and clean content
            original_sentences = []  
            processed_sentences = []  
            cleaned_process_sentences = [] 
            total_lines = len(content.splitlines())
            print(f"Total lines in content: {total_lines}")

            # Common abbreviations that shouldn't be split
            abbreviations = [
                'mr', 'mrs', 'dr', 'ms', 'inc', 'ltd', 'prof', 'sr', 'jr', 
                'st', 'mt', 'vs', 'amp', 'faq', 'etc', 'e.g', 'i.e'
            ]
            abbrev_pattern = r'\b(?:' + '|'.join(abbreviations) + r')\.$'  # Regex to match abbreviations ending with a period

            # stop words
            stop_words = {
                'the', 'is',  'of',  'a', 'on', 'for', 
                'with',  'it',  'by', 'this', 'from',  'an', 'be', 'was', 'were', 'are'
            }

            # Regex to match numbered sentences
            number_sentence_pattern = re.compile(r'(?<!\d)(\d+\.\d*)\s+(?=[A-Za-z])') 

            for i, line in enumerate(content.splitlines()):
                # Ensure each line is a string before processing
                if not isinstance(line, str):
                    line = str(line)

                
                time.sleep(0.02)
                self.progress_bar.setValue(int((i + 1) / total_lines * 100))

                # Remove non-ASCII characters
                line = re.sub(r'[^\x00-\x7F]+', '', line)

                # Ensure that numbered sentences (e.g., 1.1 sentence) are properly split into new lines
                line = re.sub(number_sentence_pattern, r'\1 ', line)  

                # Remove whitespace 
                line = re.sub(r'(?<=\d)\s*\.\s*', '.', line)

                # Custom segmentation logic for original text
                sentences = re.split(r'(?<=[.!?]) +', line)  # Split by sentence-ending punctuation

                for sentence in sentences:
                    sentence = sentence.strip()  # Clean whitespace from ends

                    # Skip if it's an abbreviation or short sentence
                    if re.search(abbrev_pattern, sentence) or len(sentence.split()) < 3:
                        continue  # Skip short sentences or known abbreviations

                    # Add a quote to sentences starting with '=' or '+' to avoid Excel issues
                    if sentence.startswith(('=', '+', '-')):
                        sentence = "'" + sentence

                    original_sentences.append(sentence)  # Store original sentences

                    # Clean up each sentence for processed output
                    # Allow numbers, periods (in decimals), and parentheses
                    cleaned_sentence = re.sub(r'[^a-zA-Z\s,.0-9()]', '', sentence)
                    if cleaned_sentence.strip():  
                        processed_sentences.append(cleaned_sentence.strip())
                    else:
                        processed_sentences.append('') 

                    # Additional cleaning for "Cleaned Process"
                    cleaned_process = re.sub(r'[^a-zA-Z\s]', '', sentence)  # Remove numbers, special chars, punctuation
                    cleaned_process = ' '.join([word for word in cleaned_process.split() if word not in stop_words and len(word) > 1])  # Manually remove stopwords

                    cleaned_process_sentences.append(cleaned_process)

            print(f"Processed sentences count: {len(processed_sentences)}")

            # Write to CSV
            csv_file = os.path.join(folder_path, "1_preprocess_output.csv")
            with open(csv_file, "w", newline='', encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Cleaned Process', 'Processed Sentence', 'Original Sentence'])  # Header row
                for cleaned, processed, original in zip(cleaned_process_sentences, processed_sentences, original_sentences):
                    if processed:  # Only write non-empty processed sentences
                        csv_writer.writerow([cleaned, processed, original])
            print(f"CSV file saved to: {csv_file}")

            # Display processed sentences on the right text area
            self.right_placeholder.setPlainText("\n".join(processed_sentences))

            # Update sentence count label
            self.sentence_count_label.setText(f"Sentence Count: {len(processed_sentences)}")

            self.progress_bar.setVisible(False)

        except Exception as e:
            
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            self.right_placeholder.setPlainText(error_message)
            self.progress_bar.setVisible(False)

    def analyze_keywords(self):
        try:
            # Check if a file has been selected
            if not hasattr(self, 'file_name') or not self.file_name:
                self.right_placeholder.setPlainText("No file selected. Please upload a file first.")
                print("Error: No file selected in analyze_keywords")
                return

            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load preprocessed sentences from CSV
            csv_file = os.path.join(folder_path, "1_preprocess_output.csv")
            if not os.path.exists(csv_file):
                self.right_placeholder.setPlainText("Preprocessed CSV file not found. Please run the Pre-process step first.")
                return

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            if df.empty:
                self.right_placeholder.setPlainText("No preprocessed sentences found. Please check the preprocessed CSV file.")
                return

            results = []
            # Analyze each "Cleaned Process" sentence for keyword occurrences
            for index, row in df.iterrows():
                original_sentence = row['Original Sentence']
                cleaned_process_sentence = row['Cleaned Process']

                # Ensure cleaned_process_sentence is a string
                if pd.isna(cleaned_process_sentence):
                    cleaned_process_sentence = ''  # Replace NaN with an empty string
                else:
                    cleaned_process_sentence = str(cleaned_process_sentence)  # Convert to string if not NaN

                # Count occurrences of keywords in the cleaned process sentence
                counts = {key: sum(cleaned_process_sentence.lower().count(keyword) for keyword in keywords) for key, keywords in self.keywords.items()}

                # Prepare the keywords found in the sentence with counts
                keywords_found = [f"{key} ({counts[key]})" for key in counts if counts[key] > 0]
                total_keywords = len(keywords_found)
                keywords_found_str = ", ".join(keywords_found) if keywords_found else "None"

                # Append results
                results.append({
                    "Total Keywords": total_keywords,
                    "Keywords Found": keywords_found_str,
                    "Cleaned Process": cleaned_process_sentence,
                    "Original Sentence": original_sentence
                })

            # Create DataFrame for results
            analysis_df = pd.DataFrame(results)

            # Save the analysis to a CSV file in the existing folder
            analysis_csv_file = os.path.join(folder_path, "2_keyword_analysis.csv")
            analysis_df.to_csv(analysis_csv_file, index=False)

            # Display results in the right placeholder
            display_text = analysis_df.to_string(index=False, col_space=10, justify='left')
            self.right_placeholder.setPlainText(f"Keyword Analysis:\n{display_text}\n\nAnalysis saved to {analysis_csv_file}")

        except Exception as e:
            # Catch any errors that happen during processing and display them
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            self.right_placeholder.setPlainText(error_message)

    def perform_nmf(self):
        try:
            # Load GloVe vectors if not already loaded
            if self.glove_vectors is None:
                self.glove_vectors = self.load_glove_vectors('600rows100d_training.txt')
                if self.glove_vectors is None:
                    self.right_placeholder.setPlainText("Error loading GloVe vectors.")
                    return

            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load preprocessed sentences from CSV
            csv_file = os.path.join(folder_path, "1_preprocess_output.csv")
            if not os.path.exists(csv_file):
                self.right_placeholder.setPlainText("Preprocessed CSV file not found. Please run the Pre-process step first.")
                return
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            if df.empty:
                self.right_placeholder.setPlainText("No preprocessed sentences found. Please check the preprocessed CSV file.")
                return

            # ===================== NMF ALONE (TF-IDF) PROCESS =====================

            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_A_matrix = tfidf_vectorizer.fit_transform(df['Cleaned Process'].fillna('')).toarray()  # Fill NaNs with empty strings
            # pd.DataFrame(tfidf_A_matrix).to_csv(os.path.join(folder_path, 'test_tf-idf_H_matrix.csv'), index=False)

            # Apply additive shift to make all values non-negative
            min_value = np.min(tfidf_A_matrix)  # Find the minimum value in the matrix
            if min_value < 0:
                tfidf_A_matrix += np.abs(min_value)  # Shift the entire matrix so the smallest value becomes 0

            # Apply NMF on TF-IDF matrix (NMF Alone)
            n_components = 30  # Number of topics
            tfidf_nmf_model = NMF(n_components=n_components, init='random', random_state=0, max_iter=3000)
            tfidf_W = tfidf_nmf_model.fit_transform(tfidf_A_matrix)
            tfidf_H = tfidf_nmf_model.components_

            # Save TF-IDF A_matrix, W, and H matrices to CSV
            pd.DataFrame(tfidf_A_matrix).to_csv(os.path.join(folder_path, '3_tf-idf_A_matrix.csv'), index=False)
            pd.DataFrame(tfidf_W).to_csv(os.path.join(folder_path, '3_tf-idf_W_matrix.csv'), index=False)
            pd.DataFrame(tfidf_H).to_csv(os.path.join(folder_path, '3_tf-idf_H_matrix.csv'), index=False)


            # ===================== NMF + GloVe PROCESS =====================
            
            # Prepare the input matrix A from GloVe vectors using the cleaned sentences
            A_matrix = []

            for cleaned_sentence in df['Cleaned Process']:
                # Ensure cleaned_sentence is a string, replace NaN with an empty string
                if pd.isna(cleaned_sentence):
                    cleaned_sentence = '' 
                else:
                    cleaned_sentence = str(cleaned_sentence)

                # Split the cleaned sentence into words
                words = cleaned_sentence.split()
                word_vectors = [self.glove_vectors[word] for word in words if word in self.glove_vectors]

                if word_vectors:
                    # Average the word vectors for the sentence
                    sentence_vector = np.mean(word_vectors, axis=0)
                    A_matrix.append(sentence_vector)
                else:
                    # If no words matched, append a zero vector
                    A_matrix.append(np.zeros(len(next(iter(self.glove_vectors.values())))))  # Use dimensionality from GloVe vectors

            A_matrix = np.array(A_matrix)

            # Check if A_matrix is empty
            if A_matrix.size == 0:
                self.right_placeholder.setPlainText("GloVe matrix (A_matrix) is empty. Please check the GloVe embeddings or input sentences.")
                return

            # Apply additive shift to make all values non-negative
            min_value = np.min(A_matrix)  # Find the minimum value in the matrix
            if min_value < 0:
                A_matrix += np.abs(min_value)  # Shift the entire matrix so the smallest value becomes 0

            # Check if the TF-IDF matrix (tfidf_W) is valid
            if tfidf_W is None or tfidf_W.size == 0:
                self.right_placeholder.setPlainText("TF-IDF matrix is empty or invalid. Please check the pre-processing step.")
                return

            # Combine the NMF (TF-IDF) W matrix with GloVe embeddings (concatenation)
            try:
                combined_matrix = np.hstack((A_matrix, tfidf_A_matrix))  # Combine GloVe embeddings with the original TF-IDF matrix
            except Exception as e:
                self.right_placeholder.setPlainText(f"Error combining matrices: {str(e)}")
                return

            # Proceed with NMF processing after validating that combined_matrix exists
            if combined_matrix.size == 0:
                self.right_placeholder.setPlainText("Combined matrix is empty. Cannot proceed with NMF.")
                return

            # Apply NMF to the combined matrix (NMF + GloVe)
            n_samples, n_features = combined_matrix.shape
            n_components = min(n_components, min(n_samples, n_features))  # Ensure valid n_components

            model = NMF(n_components=n_components, init='nndsvd', random_state=0, max_iter=1000)
            W = model.fit_transform(combined_matrix)
            H = model.components_
            
            # Save A_matrix, W, and H matrices to CSV files in the folder_path (NMF + GloVe)
            pd.DataFrame(A_matrix).to_csv(os.path.join(folder_path, '3_A_matrix.csv'), index=False)  # Save GloVe A matrix
            pd.DataFrame(W).to_csv(os.path.join(folder_path, '3_W_matrix.csv'), index=False)
            pd.DataFrame(H).to_csv(os.path.join(folder_path, '3_H_matrix.csv'), index=False)

            # Prepare to extract top words for each topic
            vocab = list(self.glove_vectors.keys())  
            n_top_words = 10  # Number of top words to display per topicW
            topics = {}
            
            for topic_idx, topic in enumerate(H):
                top_word_indices = topic.argsort()[-n_top_words:][::-1]  # Get indices of top words
                top_words = [vocab[i] for i in top_word_indices if i < len(vocab)]  # Get the actual words
                topics[f'Topic {topic_idx + 1}'] = top_words

            # Convert the topics dictionary to a DataFrame and save to CSV
            topics_df = pd.DataFrame.from_dict(topics, orient='index').transpose()
            topics_df.to_csv(os.path.join(folder_path, 'topic_words.csv'), index=False)

            # Output the top words for each topic
            output_text = "NMF Process Completed. A Matrix:\n" + pd.DataFrame(A_matrix).to_string(index=False)
            output_text += "\nTop words for each topic:\n" + topics_df.to_string(index=False)
            self.right_placeholder.setPlainText(output_text)

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(50)

            # Finalize progress
            time.sleep(1)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred during NMF processing: {str(e)}")

    def calculate_nmf_score(self):
        try:
            # Check if 'H_matrix.csv' and 'W_matrix.csv' exist in the correct directory
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # ========== Scoring for GloVe + NMF ==========
            if not os.path.exists(os.path.join(folder_path, '3_H_matrix.csv')) or not os.path.exists(os.path.join(folder_path, '3_W_matrix.csv')):
                self.right_placeholder.setPlainText("H_matrix.csv or W_matrix.csv not found. Please perform NMF first.")
                return

            # Load W and H matrices from CSV files for GloVe + NMF
            W_matrix = pd.read_csv(os.path.join(folder_path, '3_W_matrix.csv'))
            W = W_matrix.values  

            # Calculate weights for each topic (sum across sentences)
            topic_weights = np.sum(W, axis=0) / np.sum(W)  # Calculate topic weights using W matrix

            # Calculate GRS for each sentence using the rows of W matrix
            grs_scores = []
            for i in range(W.shape[0]):  # Iterate through all rows (sentences)
                grs_score = np.sum(W[i, :] * topic_weights)  # Calculate GRS for each row (sentence)
                grs_scores.append(grs_score)

            # Load the cleaned sentences from the preprocessed CSV
            preprocessed_file = os.path.join(folder_path, "1_preprocess_output.csv")
            if not os.path.exists(preprocessed_file):
                self.right_placeholder.setPlainText("Preprocessed CSV file not found. Please run the Pre-process step first.")
                return

            # Read the preprocessed CSV file into a DataFrame
            df = pd.read_csv(preprocessed_file)

            # Ensure there is a matching number of sentences
            if len(df) < len(grs_scores):
                self.right_placeholder.setPlainText("Mismatch between number of sentences and calculated scores.")
                return

            # Create a DataFrame to store scores, cleaned, and original sentences
            grs_df = pd.DataFrame({
                'Score': grs_scores,
                'Cleaned Process': df['Cleaned Process'].values[:len(grs_scores)],  # Use cleaned process
                'Original Sentence': df['Original Sentence'].values[:len(grs_scores)],  # Use original sentence
            })

            # Save to CSV in the appropriate folder
            output_csv = os.path.join(folder_path, "4_NMF_GRSScores.csv")
            grs_df.to_csv(output_csv, index=False)

            # Display the results in the right placeholder
            display_text = "GRS Scores (GloVe + NMF):\n"
            display_text += grs_df.to_string(index=False, col_space=10, justify='left')

            # Show GRS scores in the right text area and notify the user about CSV
            self.right_placeholder.setPlainText(f"{display_text}\n\nGRS scores calculated and saved to {output_csv}")

            # ========== Scoring for TF-IDF NMF ==========
            # Check if '3_tf-idf_W_matrix.csv' exists
            tfidf_w_matrix_path = os.path.join(folder_path, '3_tf-idf_W_matrix.csv')
            if not os.path.exists(tfidf_w_matrix_path):
                self.right_placeholder.setPlainText("TF-IDF W matrix not found. Please perform NMF on TF-IDF first.")
                return

            # Load the TF-IDF W matrix
            tfidf_W_matrix = pd.read_csv(tfidf_w_matrix_path)
            tfidf_W = tfidf_W_matrix.values  

            # Calculate weights for each topic (sum across sentences)
            tfidf_topic_weights = np.sum(tfidf_W, axis=0) / np.sum(tfidf_W)  # Calculate topic weights using TF-IDF W matrix

            # Calculate GRS for each sentence using the rows of TF-IDF W matrix
            tfidf_grs_scores = []
            for i in range(tfidf_W.shape[0]):  # Iterate through all rows (sentences)
                tfidf_grs_score = np.sum(tfidf_W[i, :] * tfidf_topic_weights)  # Calculate GRS for each row (sentence)
                tfidf_grs_scores.append(tfidf_grs_score)

            # Ensure there is a matching number of sentences for TF-IDF
            if len(df) < len(tfidf_grs_scores):
                self.right_placeholder.setPlainText("Mismatch between number of sentences and calculated TF-IDF scores.")
                return

            # Create a DataFrame to store TF-IDF scores, cleaned, and original sentences
            tfidf_grs_df = pd.DataFrame({
                'Score': tfidf_grs_scores,
                'Cleaned Process': df['Cleaned Process'].values[:len(tfidf_grs_scores)],  # Use cleaned process
                'Original Sentence': df['Original Sentence'].values[:len(tfidf_grs_scores)],  # Use original sentence
            })

            # Save the TF-IDF scores to a new CSV file
            tfidf_output_csv = os.path.join(folder_path, "4_TFIDF_NMF_GRSScores.csv")
            tfidf_grs_df.to_csv(tfidf_output_csv, index=False)

            # Display the TF-IDF results in the right placeholder
            display_text += "\n\nGRS Scores (TF-IDF + NMF):\n"
            display_text += tfidf_grs_df.to_string(index=False, col_space=10, justify='left')

            self.right_placeholder.setPlainText(f"{display_text}\n\nTF-IDF GRS scores calculated and saved to {tfidf_output_csv}")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred during GRS calculation: {str(e)}")

    def calculate_surface_features(self):
        try:
            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load the preprocessed sentences from the preprocess_output.csv file
            preprocessed_file = os.path.join(folder_path, "1_preprocess_output.csv")
            if not os.path.exists(preprocessed_file):
                self.right_placeholder.setPlainText("Preprocessed CSV file not found. Please run the Pre-process step first.")
                return

            # Read the CSV file into a DataFrame
            df = pd.read_csv(preprocessed_file)

            # Ensure there are sentences to process
            if df.empty:
                self.right_placeholder.setPlainText("No preprocessed sentences found. Please check the preprocessed file.")
                return

            # Replace NaN values with empty strings and ensure all values are strings
            df['Cleaned Process'] = df['Cleaned Process'].fillna('').astype(str)

            # Extract cleaned and original sentences
            cleaned_sentences = df['Cleaned Process'].tolist()
            original_sentences = df['Original Sentence'].tolist()

            position_scores = []
            length_scores = []
            final_scores = []

            # Calculate position and length scores for the preprocessed sentences
            for i, sentence in enumerate(cleaned_sentences):
                position_score = 1 / (i + 1)  # Position score (higher for earlier sentences)
                length = len(sentence.split())  # Calculate the length of the sentence (in words)

                # Calculate length score, capping it at 5 points
                if length <= 5:
                    length_score = 0  # No points for short sentences
                else:
                    length_score = min(5, (length - 5))  # Max length score of 5 for sentences longer than 5 words

                # Average the position and length scores to get the final score
                final_score = position_score + length_score

                # Append scores for the sentence
                position_scores.append(position_score)
                length_scores.append(length_score)
                final_scores.append(final_score)

            # Prepare the DataFrame with the specified order
            surface_df = pd.DataFrame({
                'Final Score': final_scores,
                'Position Score': position_scores,
                'Length Score': length_scores,
                'Cleaned Sentence': cleaned_sentences,
                'Original Sentence': original_sentences
            })

            # Save the surface feature scores to a CSV file in the appropriate folder
            output_csv = os.path.join(folder_path, "5_surface_features.csv")
            surface_df.to_csv(output_csv, index=False)

            # Display results in the right placeholder
            display_text = surface_df.to_string(index=False, col_space=10, justify='left')
            self.right_placeholder.setPlainText(f"Surface Features:\n{display_text}\n\nSurface features saved to {output_csv}")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred during surface feature calculation: {str(e)}")

    def calculate_content_features(self):
        try:
            # Load GloVe vectors if not already loaded
            if self.glove_vectors is None:
                self.glove_vectors = self.load_glove_vectors('600rows100d_training.txt')
                if self.glove_vectors is None:
                    self.right_placeholder.setPlainText("Error loading GloVe vectors.")
                    return  

            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load preprocessed sentences from preprocess_output.csv
            preprocessed_file = os.path.join(folder_path, "1_preprocess_output.csv")
            if not os.path.exists(preprocessed_file):
                self.right_placeholder.setPlainText("Preprocessed CSV file not found. Please run the Pre-process step first.")
                return

            # Read the CSV file into a DataFrame
            df = pd.read_csv(preprocessed_file)

            # Replace NaN values with empty strings and ensure all values are strings
            df['Cleaned Process'] = df['Cleaned Process'].fillna('').astype(str)

            # Extract cleaned and original sentences
            cleaned_sentences = df['Cleaned Process'].tolist()
            original_sentences = df['Original Sentence'].tolist()

            if not cleaned_sentences:
                self.right_placeholder.setPlainText("No preprocessed sentences found. Please check the preprocessed file.")
                return

            # Display loading message
            self.right_placeholder.setPlainText("Calculating content features...")

            # Calculate the document centroid by averaging word vectors for all sentences
            document_vector = np.zeros(len(next(iter(self.glove_vectors.values()))))  # GloVe vector dimensionality
            total_word_count = 0

            for sentence in cleaned_sentences:
                words = sentence.split()
                sentence_vector = np.zeros_like(document_vector)
                valid_word_count = 0

                for word in words:
                    if word in self.glove_vectors:
                        sentence_vector += self.glove_vectors[word]
                        valid_word_count += 1

                if valid_word_count > 0:
                    sentence_vector /= valid_word_count  # Average the sentence vector
                    document_vector += sentence_vector
                    total_word_count += 1

            if total_word_count > 0:
                document_vector /= total_word_count  # Average the document centroid

            # Count word frequencies across the entire text for high-frequency word score
            word_frequencies = Counter()
            for sentence in cleaned_sentences:
                word_frequencies.update(sentence.split())

            # Centroid and high-frequency word scores
            centroid_scores = []
            high_frequency_word_scores = []
            final_scores = []

            # Use a set for high-frequency words to minimize repeated checks
            high_freq_words_set = set(word_frequencies.keys())

            for sentence in cleaned_sentences:
                words = sentence.split()

                # Calculate the sentence vector and its cosine similarity with the document centroid
                sentence_vector = np.zeros_like(document_vector)
                valid_word_count = 0

                for word in words:
                    if word in self.glove_vectors:
                        sentence_vector += self.glove_vectors[word]
                        valid_word_count += 1

                if valid_word_count > 0:
                    sentence_vector /= valid_word_count  # Average sentence vector

                # Normalize centroid score by sentence length
                centroid_score = cosine_similarity([sentence_vector], [document_vector])[0][0] / (len(words) if len(words) > 0 else 1)

                # Updated high-frequency word score
                high_frequency_score = 0
                if len(words) > 0:
                    for word in words:
                        if word in word_frequencies:
                            word_frequency = word_frequencies[word]

                            # Get the GloVe vector for the word if it exists
                            if word in self.glove_vectors:
                                word_vector = self.glove_vectors[word]

                                # Find semantic similarity with top 10 high-frequency words
                                similar_word_score = sum(
                                    cosine_similarity([word_vector], [self.glove_vectors[high_freq_word[0]]])[0][0] * high_freq_word[1]
                                    for high_freq_word in word_frequencies.most_common(10) if high_freq_word[0] in self.glove_vectors
                                )

                                # Combine frequency and semantic similarity
                                high_frequency_score += (word_frequency + similar_word_score) / len(words)

                high_frequency_word_scores.append(high_frequency_score)
                centroid_scores.append(centroid_score)

            # Normalize the high-frequency word scores
            max_high_frequency_score = max(high_frequency_word_scores) if high_frequency_word_scores else 1
            if max_high_frequency_score > 0:
                high_frequency_word_scores = [score / max_high_frequency_score * 5 for score in high_frequency_word_scores]

            # Compute the final score as a weighted average
            final_scores = [(centroid_score + high_frequency_score) 
                            for centroid_score, high_frequency_score in zip(centroid_scores, high_frequency_word_scores)]

            # Prepare the DataFrame with the specified order
            content_df = pd.DataFrame({
                'Final Score': final_scores,
                'Centroid Score': centroid_scores,
                'High Frequency Word Score': high_frequency_word_scores,
                'Cleaned Sentence': cleaned_sentences,
                'Original Sentence': original_sentences
            })

            # Save to CSV in the appropriate folder
            output_csv = os.path.join(folder_path, "5_content_features.csv")
            content_df.to_csv(output_csv, index=False)

            # Display results in the right placeholder
            display_text = content_df.to_string(index=False, col_space=10, justify='left')
            self.right_placeholder.setPlainText(f"Content Features:\n{display_text}\n\nContent features saved to {output_csv}")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred while calculating content features: {str(e)}")
 
    def calculate_rhetorical_features(self):
        try:
            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load preprocessed sentences from preprocess_output.csv
            preprocessed_file = os.path.join(folder_path, "1_preprocess_output.csv")
            if not os.path.exists(preprocessed_file):
                self.right_placeholder.setPlainText("Preprocessed CSV file not found. Please run the Pre-process step first.")
                return

            # Read the CSV file into a DataFrame
            df = pd.read_csv(preprocessed_file)

            # Replace NaN values with empty strings and ensure all values are strings
            df['Cleaned Process'] = df['Cleaned Process'].fillna('').astype(str)

            # Extract cleaned and original sentences
            cleaned_sentences = df['Cleaned Process'].tolist()
            original_sentences = df['Original Sentence'].tolist()

            if not cleaned_sentences:
                self.right_placeholder.setPlainText("No preprocessed sentences found. Please check the preprocessed file.")
                return

            # Combine all keywords into a single list for counting
            all_keywords = [keyword for sublist in self.keywords.values() for keyword in sublist]

            results = []
            # Analyze each cleaned sentence for keyword occurrences and calculate scores
            for i, sentence in enumerate(cleaned_sentences):
                # Split the sentence into individual words
                words = sentence.split()

                # Count the occurrences of keywords in the sentence
                keyword_count = sum(1 for word in words if word in all_keywords)

                # The final score is simply the number of keywords in the sentence
                final_score = keyword_count

                results.append({
                    "Final Score": final_score,
                    "Total Keywords Count": keyword_count,
                    "Cleaned Sentence": sentence,
                    "Original Sentence": original_sentences[i]
                })

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Save to CSV in the appropriate folder
            output_csv = os.path.join(folder_path, "5_rhetorical_features.csv")
            results_df.to_csv(output_csv, index=False)

            # Display results in the right placeholder
            display_text = results_df.to_string(index=False, col_space=10, justify='left')
            self.right_placeholder.setPlainText(f"Rhetorical Features:\n{display_text}\n\nRhetorical features saved to {output_csv}")

        except Exception as e:
            self.right_placeholder.setPlainText(f"Error calculating rhetorical features: {str(e)}")

    def calculate_feature_score(self):
        try:
            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load the results from the surface, content, and rhetorical features CSV files
            surface_df = pd.read_csv(os.path.join(folder_path, "5_surface_features.csv"))
            content_df = pd.read_csv(os.path.join(folder_path, "5_content_features.csv"))
            rhetorical_df = pd.read_csv(os.path.join(folder_path, "5_rhetorical_features.csv"))

            # Load the preprocessed sentences from the preprocess_output.csv file
            preprocessed_file = os.path.join(folder_path, "1_preprocess_output.csv")
            df = pd.read_csv(preprocessed_file)

            # Ensure the dataframes have the same number of rows
            if not (len(surface_df) == len(content_df) == len(rhetorical_df) == len(df)):
                self.right_placeholder.setPlainText("Mismatch in sentence counts between feature scores.")
                return

            # Combine the scores into a single dataframe
            combined_df = pd.DataFrame({
                'Surface Score': surface_df['Final Score'],
                'Content Score': content_df['Final Score'],
                'Rhetorical Score': rhetorical_df['Final Score'],
                'Cleaned Sentence': df['Cleaned Process'],
                'Original Sentence': df['Original Sentence']
            })

            # Calculate the overall feature score as the sum of the three scores
            combined_df['Overall Feature Score'] = combined_df[['Surface Score', 'Content Score', 'Rhetorical Score']].sum(axis=1)

            # Reorder columns to place 'Overall Feature Score' first
            combined_df = combined_df[['Overall Feature Score', 'Surface Score', 'Content Score', 'Rhetorical Score', 'Cleaned Sentence', 'Original Sentence']]

            # Save the combined results to a new CSV file
            output_csv = os.path.join(folder_path, "6_feature_scores.csv")
            combined_df.to_csv(output_csv, index=False)

            # Display results in the right placeholder
            display_text = combined_df.to_string(index=False, col_space=10, justify='left')
            self.right_placeholder.setPlainText(f"Feature Scores:\n{display_text}\n\nFeature scores saved to {output_csv}")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred while calculating feature scores: {str(e)}")
   
    def extract_sentences(self):
        try:
            # Create the folder path based on the input file
            input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
            folder_path = os.path.join("Summary", input_filename)

            # Load the necessary score dataframes
            nmf_grs_df = pd.read_csv(os.path.join(folder_path, "4_NMF_GRSScores.csv"))
            feature_df = pd.read_csv(os.path.join(folder_path, "6_feature_scores.csv"))
            processed_df = pd.read_csv(os.path.join(folder_path, "1_preprocess_output.csv"))
            tf_idf_w_df = pd.read_csv(os.path.join(folder_path, "4_TFIDF_NMF_GRSScores.csv"))
            
            # Debug prints after loading CSVs
            print(f"NMF GRS Scores CSV rows: {len(nmf_grs_df)}")
            print(f"Feature Scores CSV rows: {len(feature_df)}")
            print(f"Processed Output CSV rows: {len(processed_df)}")
            print(f"TFIDF NMF Output CSV rows: {len(tf_idf_w_df)}")

            # Check for expected columns
            print("NMF GRS Scores Columns:", nmf_grs_df.columns)
            print("Feature Scores Columns:", feature_df.columns)
            print("Processed Output Columns:", processed_df.columns)
            print("TFIDF NMF Output Columns:", tf_idf_w_df.columns)

        except FileNotFoundError as e:
            self.right_placeholder.setPlainText(f"Required files not found: {str(e)}. Please make sure the NMF and Feature Scores have been calculated.")
            return
        except Exception as e:
            self.right_placeholder.setPlainText(f"Error loading files: {str(e)}")
            return

        try:
            # Ensure both dataframes have the same number of sentences
            if len(nmf_grs_df) != len(feature_df):
                self.right_placeholder.setPlainText("Mismatch in the number of sentences between NMF GRS Scores and Feature Scores.")
                return
            
            if len(tf_idf_w_df) != len(feature_df):
                self.right_placeholder.setPlainText("Mismatch in the number of sentences between TF-IDF W matrix and Feature Scores.")
                return
                
            # Check if the number of rows matches across all three dataframes
            if len(nmf_grs_df) != len(processed_df):
                self.right_placeholder.setPlainText("Mismatch in the number of sentences between NMF GRS Scores and Preprocessed Output.")
                return

            # Use cleaned sentences from the processed DataFrame instead of NMF GRS sentences
            try:
                cleaned_sentences = processed_df['Cleaned Process']  # Extract the cleaned sentences
            except KeyError as e:
                self.right_placeholder.setPlainText(f"KeyError: Column 'Cleaned Process' not found in processed output. Please check the CSV structure.")
                return
            except Exception as e:
                self.right_placeholder.setPlainText(f"An error occurred while extracting cleaned sentences: {str(e)}")
                return

            # Create a new DataFrame using scores from both dataframes and adding original sentences
            combined_df = pd.DataFrame({
                'Sentence': cleaned_sentences,  # Use cleaned sentences
                'Original Sentence': processed_df['Original Sentence'],  # Add the original sentence column
                'NMF GRS Score': nmf_grs_df['Score'],
                'Overall Feature Score': feature_df['Overall Feature Score'],
                'Surface Score': feature_df['Surface Score'],
                'Content Score': feature_df['Content Score'],
                'Rhetorical Score': feature_df['Rhetorical Score'],
                'NMF Score': tf_idf_w_df['Score']
                
            })
            
            # NMF only using the TF-IDF W matrix scores
            nmf_only_df = combined_df[['Sentence', 'Original Sentence', 'NMF Score']].copy()  # Copy only the Sentence and Original Sentence
            nmf_only_df.loc[:, 'Total Score'] = nmf_only_df['NMF Score'] 
            nmf_only_df.loc[:, 'Rank'] = nmf_only_df['Total Score'].rank(ascending=False, method='dense').astype(int)

            nmf_only_df.to_csv(os.path.join(folder_path, "nmf.csv"), index=False)


            # Save individual CSV files for different combinations, calculating Total Score appropriately
            # NMF + GLOVE
            nmf_glove_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score']]
            nmf_glove_df.loc[:, 'Total Score'] = nmf_glove_df['NMF GRS Score']
            nmf_glove_df.loc[:, 'Rank'] = nmf_glove_df['Total Score'].rank(ascending=False, method='dense').astype(int)

            nmf_glove_df.to_csv(os.path.join(folder_path, "nmf+glove.csv"), index=False)

            # NMF + GLOVE + RHETORICAL
            nmf_glove_rhetorical_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Rhetorical Score']]
            nmf_glove_rhetorical_df.loc[:, 'Total Score'] = nmf_glove_rhetorical_df['NMF GRS Score'] + nmf_glove_rhetorical_df['Rhetorical Score']
            nmf_glove_rhetorical_df.loc[:, 'Rank'] = nmf_glove_rhetorical_df['Total Score'].rank(ascending=False, method='dense').astype(int)

            nmf_glove_rhetorical_df.to_csv(os.path.join(folder_path, "nmf+glove+rhetorical.csv"), index=False)

            # NMF + GLOVE + SURFACE
            nmf_glove_surface_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Surface Score']]
            nmf_glove_surface_df.loc[:, 'Total Score'] = nmf_glove_surface_df['NMF GRS Score'] + nmf_glove_surface_df['Surface Score']
            nmf_glove_surface_df.loc[:, 'Rank'] = nmf_glove_surface_df['Total Score'].rank(ascending=False, method='dense').astype(int)

            nmf_glove_surface_df.to_csv(os.path.join(folder_path, "nmf+glove+surface.csv"), index=False)

            # NMF + GLOVE + CONTENT
            nmf_glove_content_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Content Score']]
            nmf_glove_content_df.loc[:, 'Total Score'] = nmf_glove_content_df['NMF GRS Score'] + nmf_glove_content_df['Content Score']
            nmf_glove_content_df.loc[:, 'Rank'] = nmf_glove_content_df['Total Score'].rank(ascending=False, method='dense').astype(int)

            nmf_glove_content_df.to_csv(os.path.join(folder_path, "nmf+glove+content.csv"), index=False)

            # NMF + GLOVE + SURFACE + CONTENT
            nmf_glove_surface_content_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Surface Score', 'Content Score']]
            nmf_glove_surface_content_df.loc[:,'Total Score'] = (nmf_glove_surface_content_df['NMF GRS Score'] +
                                                        nmf_glove_surface_content_df['Surface Score'] +
                                                        nmf_glove_surface_content_df['Content Score'])
            nmf_glove_surface_content_df.loc[:,'Rank'] = nmf_glove_surface_content_df['Total Score'].rank(ascending=False, method='dense').astype(int)
            nmf_glove_surface_content_df.to_csv(os.path.join(folder_path, "nmf+glove+surface+content.csv"), index=False)

            # NMF + GLOVE + RHETORICAL + CONTENT
            nmf_glove_rhetorical_content_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Rhetorical Score', 'Content Score']]
            nmf_glove_rhetorical_content_df.loc[:,'Total Score'] = (nmf_glove_rhetorical_content_df['NMF GRS Score'] +
                                                            nmf_glove_rhetorical_content_df['Rhetorical Score'] +
                                                            nmf_glove_rhetorical_content_df['Content Score'])
            nmf_glove_rhetorical_content_df.loc[:,'Rank'] = nmf_glove_rhetorical_content_df['Total Score'].rank(ascending=False, method='dense').astype(int)
            nmf_glove_rhetorical_content_df.to_csv(os.path.join(folder_path, "nmf+glove+rhetorical+content.csv"), index=False)

            # NMF + GLOVE + RHETORICAL + SURFACE
            nmf_glove_rhetorical_surface_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Rhetorical Score', 'Surface Score']]
            nmf_glove_rhetorical_surface_df.loc[:,'Total Score'] = (nmf_glove_rhetorical_surface_df['NMF GRS Score'] +
                                                            nmf_glove_rhetorical_surface_df['Rhetorical Score'] +
                                                            nmf_glove_rhetorical_surface_df['Surface Score'])
            nmf_glove_rhetorical_surface_df.loc[:,'Rank'] = nmf_glove_rhetorical_surface_df['Total Score'].rank(ascending=False, method='dense').astype(int)
            nmf_glove_rhetorical_surface_df.to_csv(os.path.join(folder_path, "nmf+glove+rhetorical+surface.csv"), index=False)
            
            # NMF + RHETORICAL + SURFACE + CONTENT
            nmf_rhetorical_surface_content_df = combined_df[['Sentence', 'Original Sentence', 'Rhetorical Score', 'Surface Score', 'Content Score', 'NMF Score']]
            nmf_rhetorical_surface_content_df.loc[:, 'Total Score'] = (nmf_rhetorical_surface_content_df['NMF Score'] +
                                                                    nmf_rhetorical_surface_content_df['Rhetorical Score'] +
                                                                    nmf_rhetorical_surface_content_df['Surface Score'] +
                                                                    nmf_rhetorical_surface_content_df['Content Score'])
            nmf_rhetorical_surface_content_df.loc[:, 'Rank'] = nmf_rhetorical_surface_content_df['Total Score'].rank(ascending=False, method='dense').astype(int)

            nmf_rhetorical_surface_content_df.to_csv(os.path.join(folder_path, "nmf+rhetorical+surface+content.csv"), index=False)

            # Save the DataFrame with the original order and the newly added rank and original sentence for the overall results
            output_csv = os.path.join(folder_path, "7_sentence_extraction_results.csv")
            combined_df = combined_df[['Sentence', 'Original Sentence', 'NMF GRS Score', 'Overall Feature Score',
                                        'Surface Score', 'Content Score', 'Rhetorical Score']]
            combined_df['Overall Score'] = combined_df['NMF GRS Score'] + combined_df['Overall Feature Score']
            combined_df['Rank'] = combined_df['Overall Score'].rank(ascending=False, method='dense').astype(int)
            combined_df.to_csv(output_csv, index=False)

            # Display the results in the right text area in the original order
            display_text = combined_df.to_string(index=False, col_space=10, justify='left')
            self.right_placeholder.setPlainText(f"Sentence Extraction Results (Original Order):\n{display_text}\n\nResults saved to {output_csv}")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred while extracting sentences: {str(e)}")

    def display_final_sentences(self):
        # Create the folder path based on the input file
        input_filename = os.path.splitext(os.path.basename(self.file_name))[0]
        folder_path = os.path.join("Summary", input_filename)

        # Path to the extraction file
        extraction_file = os.path.join(folder_path, "7_sentence_extraction_results.csv")
        if not os.path.exists(extraction_file):
            self.right_placeholder.setPlainText("Sentence extraction results not found. Please run the summarization pipeline first.")
            return

        # Read the extracted sentences and their ranks
        extraction_df = pd.read_csv(extraction_file)

        # Calculate the number of sentences to include in the summary (top 1/3)
        top_1_3_count = len(extraction_df) // 3

        top_sentences = []
        final_summary_data = []  # List to hold data for final summary CSV
        for index, row in extraction_df.iterrows():
            rank = row['Rank']
            sentence = row['Original Sentence']

            # Check if the sentence's rank is in the top 1/3 count
            if rank <= top_1_3_count:
                top_sentences.append(sentence)
                final_summary_data.append({"Rank": rank, "Sentence": sentence})  # Append rank and sentence

        # Display the selected sentences in the right placeholder
        self.right_placeholder.setPlainText("\n".join(top_sentences))

        # Create the ALL FINAL SUMMARY folder if it doesn't exist
        all_final_summary_folder = os.path.join(os.path.dirname(folder_path), "ALL A NMF GLOVE RHETORICAL SURFACE CONTENT")
        os.makedirs(all_final_summary_folder, exist_ok=True)

        # Save the selected sentences to a text file named based on the original filename
        final_summary_filename = f"{input_filename}.txt"
        final_summary_file_path = os.path.join(all_final_summary_folder, final_summary_filename)

        with open(final_summary_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(top_sentences))

        # Save the final summary data to a new CSV file
        final_summary_df = pd.DataFrame(final_summary_data)
        final_summary_csv_path = os.path.join(folder_path, f"{input_filename}.csv")
        final_summary_df.to_csv(final_summary_csv_path, index=False)

        # Dictionary to map output folder names to the correct CSV filenames
        csv_file_mapping = {
            'ALL 1 NMF': 'nmf.csv',
            'ALL 2 NMF GLOVE': 'nmf+glove.csv',
            'ALL 3 NMF GLOVE RHETORICAL': 'nmf+glove+rhetorical.csv',
            'ALL 4 NMF GLOVE SURFACE': 'nmf+glove+surface.csv',
            'ALL 5 NMF GLOVE CONTENT': 'nmf+glove+content.csv',
            'ALL 6 NMF GLOVE SURFACE CONTENT': 'nmf+glove+surface+content.csv',
            'ALL 7 NMF GLOVE RHETORICAL CONTENT': 'nmf+glove+rhetorical+content.csv',
            'ALL 8 NMF GLOVE RHETORICAL SURFACE': 'nmf+glove+rhetorical+surface.csv',
            'ALL 9 NMF RHETORICAL SURFACE CONTENT': 'nmf+rhetorical+surface+content.csv'
        }

        try:
            # Iterate through the CSV file mappings
            for folder_name, csv_filename in csv_file_mapping.items():
                # Create a folder for the current CSV inside the Summary folder
                output_folder = os.path.join("Summary", folder_name)
                os.makedirs(output_folder, exist_ok=True)

                # Read the current CSV file
                csv_path = os.path.join(folder_path, csv_filename)
                df = pd.read_csv(csv_path)

                # Get the top 1/3 sentences based on rank
                top_1_3_df = df[df['Rank'] <= top_1_3_count]

                # Save the top 1/3 sentences to a new CSV file in the corresponding folder
                output_csv_path = os.path.join(output_folder, f"{input_filename}.csv")  # Use input filename for final output
                top_1_3_df.to_csv(output_csv_path, index=False)

                # Save the top 1/3 sentences to a text file without rank
                output_txt_path = os.path.join(output_folder, f"{input_filename}.txt")  # Create a txt file with the same name
                with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write("\n".join(top_1_3_df['Original Sentence'].tolist()))  # Write only sentences

                print(f"Top 1/3 sentences saved to {output_csv_path}")
                print(f"Top 1/3 sentences saved to {output_txt_path}")  # Print path of txt file

        except Exception as e:
            self.right_placeholder.setPlainText(f"Error while saving top 1/3 sentences: {str(e)}")

        print(f"Final summary saved to {final_summary_csv_path}")
        print(f"Final summary text file saved to {final_summary_file_path}")

    def batch_summarize_documents(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting batch summarization...")

        try:
            # Ensure files were selected in the batch upload dialog
            if hasattr(self, 'file_list') and self.file_list:
                total_files = len(self.file_list)
                for idx, file_path in enumerate(self.file_list):
                    # Update progress for each file
                    self.progress_label.setText(f"Processing file {idx + 1}/{total_files}: {os.path.basename(file_path)}")

                    # Display the content of the file in the left part of the body
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        self.left_text_area.setPlainText(file_content)  # Assuming left_placeholder is where you display content

                    # Allow the UI to update before processing the file
                    QApplication.processEvents()

                    # Set the current file to be processed and call the summarization function
                    self.file_name = file_path
                    self.summarize_document()  # Process the current file

                    # Update progress bar for each file
                    progress_value = int(((idx + 1) / total_files) * 100)
                    self.progress_bar.setValue(progress_value)
            else:
                self.progress_label.setText("No files were selected for batch summarization.")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred during batch summarization: {str(e)}")
            self.progress_label.setText("Batch summarization failed. See the output for details.")
        
        finally:
            self.progress_label.setText("Batch summarization completed.")
            self.progress_bar.setVisible(False)
    
    def summarize_document(self):
        # Reset the progress bar and the placeholder message
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting the summarization process...")

        try:
            # Step 1: Pre-process the file
            self.progress_label.setText("Step 1: Pre-processing the file...")
            self.preprocess_file()
            self.progress_bar.setValue(10)
            
            # Step 2: Analyze keywords
            self.progress_label.setText("Step 2: Analyzing keywords...")
            self.analyze_keywords()
            self.progress_bar.setValue(20)

            # Step 3: Perform NMF
            self.progress_label.setText("Step 3: Performing NMF...")
            self.perform_nmf()
            self.progress_bar.setValue(30)

            # Step 4: Calculate NMF Score
            self.progress_label.setText("Step 4: Calculating NMF score...")
            self.calculate_nmf_score()
            self.progress_bar.setValue(40)

            # Step 5: Calculate Surface Features
            self.progress_label.setText("Step 5: Calculating Surface Features...")
            self.calculate_surface_features()
            self.progress_bar.setValue(50)

            # Step 6: Calculate Content Features
            self.progress_label.setText("Step 6: Calculating Content Features...")
            self.calculate_content_features()
            self.progress_bar.setValue(60)

            # Step 7: Calculate Rhetorical Features
            self.progress_label.setText("Step 7: Calculating Rhetorical Features...")
            self.calculate_rhetorical_features()
            self.progress_bar.setValue(70)

            # Step 8: Calculate Feature Score
            self.progress_label.setText("Step 8: Calculating Feature Score...")
            self.calculate_feature_score()
            self.progress_bar.setValue(80)

            # Step 9: Extract Sentences
            self.progress_label.setText("Step 9: Extracting Sentences...")
            self.extract_sentences()
            self.progress_bar.setValue(100)

            # Indicate completion
            self.progress_label.setText("Summarization process completed successfully!")

        except Exception as e:
            self.right_placeholder.setPlainText(f"An error occurred during summarization: {str(e)}")
            self.progress_label.setText("Summarization failed. See the output for details.")
        
        finally:
            # Hide the progress bar once the process is complete
            time.sleep(0.5)
            self.progress_bar.setVisible(False)
            
            # Step 10: Display the Top 50% Sentences Based on Rank
            self.display_final_sentences()  # Change this line





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernUI()
    window.show()
    sys.exit(app.exec_())
