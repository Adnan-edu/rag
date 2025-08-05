# 🍽️ Restaurant Menu Automation Project: From Images to Excel with GenAI ✨

This project offers an innovative solution to automate the conversion of restaurant menus from images or PDFs into structured Excel files, significantly reducing the need for manual data entry. This free service is designed to help restaurants quickly generate upload-ready spreadsheets, thereby saving time and costs. The primary goal is to transform unstructured menu data into a clean, consistent, and usable tabular format for further analysis or integration.

## 🚀 Project Overview & Key Objectives

The inspiration for this project came from the real-world challenge faced by a startup where menu items had to be added one by one, including translations, which was highly inefficient, especially for menus with hundreds of items. Leveraging vision models and Generative AI (GenAI), this solution automates that process.

**Key Objectives:**
*   **Automate Data Entry:** Eliminate the manual and time-consuming process of entering menu items and their details.
*   **Cost Savings:** Provide a free service that helps restaurants save money they might otherwise spend on digital menu services (e.g., €480 per year).
*   **Structured Output:** Convert diverse menu formats (PDFs, images) into a standardized Excel spreadsheet that adheres to a strict template.
*   **Scalability:** Develop a process that can efficiently handle multiple PDFs and images to generate a comprehensive menu Excel file.

## 💡 The Workflow: PDF → Image → Excel

The entire process is broken down into two main stages, each handled by a dedicated Jupyter Notebook:

1.  **PDF to Images Conversion** (using `PDF to Images.ipynb`) ➡️ Converts PDF menus into individual image files.
2.  **Image to Excel Conversion with GenAI** (using `Image to Excel GenAI.ipynb`) ➡️ Processes these images (or any other menu images) and extracts structured data into an Excel spreadsheet.

This streamlined workflow allows for the creation of an upload-ready Excel file in less than 5 minutes.

## 📂 Step 1: `PDF to Images.ipynb` 📸

This notebook is responsible for converting PDF menu files into image formats (JPG). This is a crucial preliminary step when the initial menu source is a PDF, as the subsequent GenAI model primarily works with images.

**How it works:**
*   The notebook utilizes libraries like `fitz` (PyMuPDF) and `PIL` (Pillow) to handle PDF and image processing.
*   It identifies all `.pdf` files within a specified source directory.
*   For each PDF document, it iterates through its pages.
*   Each page is converted into a pixmap and then saved as a separate JPG image file.
*   Existing contents in the target directory are removed before saving new images to ensure a clean conversion.
*   The output images are stored in a designated `pdf_to_image` directory.

## 📊 Step 2: `Image to Excel GenAI.ipynb` 🤖

This is the core of the project, where **Generative AI** is used to analyze menu images and convert their content into a structured Excel format.

**Key Components & Workflow:**

1.  **AI Model & System Prompt:**
    *   The project uses the **`gpt-4o`** model from OpenAI for its analysis and conversion capabilities.
    *   A **detailed system prompt** instructs the `gpt-4o` model on how to interpret the menu images and convert them into the required structured Excel format. This prompt ensures data consistency and adherence to a specific template.

2.  **Excel Template Structure (Columns Guide):**
    The output Excel spreadsheet follows a precise structure, with each row representing a unique menu item. Category and subcategory names are repeated for items within the same subcategory, and certain columns are left blank if not applicable.
    The strict column definitions ensure database integrity and proper data representation

3.  **Core Workflow Phases:**
    The process of extracting structured data from a set of menu images involves these phases:

    *   **Phase 1: Image Retrieval & Encoding** 🖼️
        *   For each image file (PNG, JPG, JPEG) in the specified directory, the image is loaded.
        *   The image is then encoded into **base64 format** for transmission to the AI model.

    *   **Phase 2: GenAI Model Analysis** 🧠
        *   The encoded image is sent to the `gpt-4o` model.
        *   A user prompt instructs the model to "Convert this menu image to a structured Excel Sheet Format".

    *   **Phase 3: Data Extraction & Aggregation** 📥
        *   The model's response, typically formatted as a markdown table, is parsed row by row.
        *   Each valid row of extracted data is then appended to a **Pandas DataFrame**. This DataFrame gradually builds a comprehensive structured dataset from all processed images.

4.  **Output:**
    *   The final Pandas DataFrame, containing all the extracted and structured menu data, is then saved as an **Excel file (`.xlsx`)**.
    *   The user is prompted to enter a desired filename for the Excel output.

This automated process ensures that restaurant menus, whether from PDFs or images, are quickly and accurately transformed into a usable, structured format, ready for any digital menu system or further analysis.