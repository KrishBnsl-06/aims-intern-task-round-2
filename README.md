# AIMS Intern Task Round 2

This project leverages the Gemma model to generate concise and accurate recipes based on a given dish image and its brief description. It was developed as part of the AIMS Internship Program.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Sample Output](#sample-output)
* [License](#license)

## Overview

The application processes an image of a dish along with a vague textual description to produce a precise recipe. It utilizes machine learning techniques to interpret visual and textual inputs, generating structured recipe outputs.

## Features

* Image analysis to identify dish components.
* Text processing to comprehend vague descriptions.
* Integration of visual and textual data for recipe generation.
* Output of structured recipes including ingredients and preparation steps.([GitHub][1])

## Project Structure

```

├── aims-intern-round2.py   # Main application script
├── recipes_2.csv           # Dataset containing sample recipes
├── images/                 # Directory with sample dish images
├── shell.nix               # Nix shell configuration for environment setup
└── README.md               # Project documentation
```



## Installation

### Prerequisites

* Python 3.8 or higher
* Git
* Nix (optional, for environment setup)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/KrishBnsl-06/aims-intern-task-round-2.git
   cd aims-intern-task-round-2
   ```



2. **Set Up the Environment**

   * **Using Nix:**

     ```bash
     nix-shell shell.nix
     ```

   * **Using pip:**

     ```bash
     pip install -r requirements.txt
     ```

## Usage

1. **Prepare Input**

   * Place the dish image in the `images/test-images` directory.
   * Ensure the image filename and corresponding description are correctly referenced in the script.

2. **Run the Application**

   ```bash
   python aims-intern-round2.py
   ```



3. **View Output**

   * The generated recipe will be displayed in the console or saved to an output file, depending on the script's configuration.