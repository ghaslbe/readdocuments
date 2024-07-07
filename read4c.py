import os
import sys
import re
import cv2
import pytesseract
from PIL import Image
import numpy as np
from pyzbar import pyzbar
import subprocess
import json

def list_available_languages():
    try:
        languages = pytesseract.get_languages(config='')
        # print("Available languages for Tesseract OCR:")
        # for lang in languages:
            # print(lang)
    except Exception as e:
        print(f"Error listing languages: {e}")

def preprocess_image(image_path):
    # Bild mit OpenCV laden
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Bild skalieren
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rauschunterdrückung
    gray = cv2.medianBlur(gray, 3)

    # Barcodes erkennen und entfernen
    barcodes = pyzbar.decode(gray)
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), -1)

    # Binarisierung
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Das vorverarbeitete Bild speichern
    cv2.imwrite('tmp.jpg', binary_image)

    return binary_image

def extract_text_from_image(image_path):
    try:
        # Bild vorverarbeiten
        preprocessed_image = preprocess_image(image_path)

        # Bild in eine PIL-Image umwandeln
        pil_image = Image.fromarray(preprocessed_image)

        # Tesseract-Konfiguration anpassen (Deutsch als Sprache)
        custom_config = r'--oem 3 --psm 6 -l deu'

        # Text mit Tesseract extrahieren
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        pattern1 = r'(?<=\n)[^\w\s]+(?=\s)'  # Für nicht-alphanumerische Zeichen
        pattern2 = r'(?<=\n)\S{1,2}(?=\s)'   # Für Zeichenketten von 1-2 Zeichen

        # Anwendung der Muster auf den Text
        cleaned_text = re.sub(pattern1, '', text)
        cleaned_text = re.sub(pattern2, '', cleaned_text)

        # Anwendung des Musters auf den Text
        text = cleaned_text

        return text
    except Exception as e:
        return str(e)

def call_ollama(prompt):
    #print("Rufe Ollama auf...")
    try:
        # result = subprocess.run(['ollama', 'run', 'gemma2:latest', prompt], capture_output=True, text=True)
        result = subprocess.run(['ollama', 'run', 'gemma2:latest', ' --temperature 0.1', prompt], capture_output=True, text=True)
  
        return result.stdout
    except Exception as e:
        return str(e)

def analyze_text(extracted_text):
    prompt = f"""
    was fällt dir an folgendem text auf? sind die daten in sich stimmig. Prüfe die einzelnen angaben. Stimmen zum beispiel land des Empfängers und Iban überein? was fällt dir auf ? Was ist nicht stimmig, was könnte eine drohung sein?
    Ignoriere Dinge wie fehlende Umlaute, Zahlen oder Abkürzungen ausser sie deuten auf Missbrauch oder drohungen hin.
    Erstelle deine ausgaben Nur als json mit 2 Bereichen. Gib keinerlei andere Daten aus, ausser JSON.
    Seltsam: Hier ist eine Liste von auffälligkeiten hinterlegt
    Scorewert: 1: unauffällig, 100: auffällig
    Starte mit ```JSON und ende mit ```
    Gib immer ein JSON Objekt aus.
    Hier ist der Text: 
    {extracted_text}
    """
    return call_ollama(prompt)

def identify_document_type(extracted_text):
    prompt = f"""
    Erkenne, ob es sich um eines der folgenden Dokumente handelt.
    Gib den Typ als Json zurück. Gib keinerlei andere Daten aus, ausser die JSON Daten. 
    Starte mit ```JSON und ende mit ```
    Gib immer ein JSON Objekt aus.
    Hier ist der Text: 

    Hier die Kategorien dazu:
    1. Versicherungsnummer
    2. Versicherungssparte
    3. Kontaktanlass
    4. Name
    5. Adresse
    6. E-Mail
    7. Geschäftsvorfall, aus nachfolgenden Kategorien und Unterkategorien:
    Versicherungssparte
    * Kfz
    * Hausrat
    * Wohngebäude
    * Haftpflicht
    * Lebensversicherung
    * Krankenversicherung
    Schadenmeldung
    * Kfz-Schäden
    * Hausrat- und Wohngebäudeschäden
    * Haftpflichtschäden
    * Personenschäden
    Vertragsangelegenheiten
    * Vertragsabschluss
    * Vertragsänderung
    * Vertragskündigung
    * Vertragsverlängerung
    Leistungsanfragen
    * Leistungsanträge
    * Leistungsabrechnungen
    * Rückfragen zu Leistungen
    Rechnungen und Zahlungen
    * Beitragszahlungen
    * Erstattungsanfragen
    * Mahnungen
    * Zahlungseingänge
    Beschwerden und Reklamationen
    * Leistungsbeschwerden
    * Servicebeschwerden
    * Schadenregulierungsbeschwerden
    Informationen und Anfragen
    * Produktinformationen
    * Beratungsgesuche
    * allgemeine Anfragen
    Dokumentenanforderungen
    * Kopien von Verträgen
    * Nachweise und Bescheinigungen
    * Gutachtenanforderungen
    Risikomeldungen und Anderungsanzeigen
    * Risikoänderungen (z.B. Wohnortwechsel, Fahrzeugwechsel)
    * Berufliche Änderungen
    * Persönliche Änderungen (z.B. Eheschließung, Geburt)
    Marketing und Angebote
    * Werbemitteilungen
    * Sonderangebote
    * Kundenumfragen
    Sonstiges
    * Unkategorisierte Anliegen
    * Allgemeine Korrespondenz
    
    {extracted_text}
    """
    return call_ollama(prompt)

if __name__ == "__main__":
    # Verfügbare Sprachen auflisten
    list_available_languages()

    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    extracted_text = extract_text_from_image(image_path)

    # Analyse Text
    analysis_result = analyze_text(extracted_text)
    print(analysis_result)

    # Dokumententyp erkennen
    document_type_result = identify_document_type(extracted_text)
    print(document_type_result)

