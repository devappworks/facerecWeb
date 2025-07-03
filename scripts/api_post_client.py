
"""
API GET Client Script
Jednostavna skripta za pozivanje GET API endpoint-a sa procesiranjem članaka
"""

import requests
import json
import sys
import argparse
import os
import re
from typing import Dict, Any, Set, List
from datetime import datetime
import pandas as pd

# Dodajemo putanju do app modula
app_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(app_path)
print(f"🔧 Dodata putanja: {app_path}")


def sanitize_domain_for_filename(domain: str) -> str:
    """Konvertuje domain u sigurno ime fajla"""
    # Uklanjamo protokol (http://, https://)
    domain = re.sub(r'^https?://', '', domain)
    # Zamenjujemo tačke i specijalne karaktere sa underscore
    domain = re.sub(r'[^a-zA-Z0-9]', '_', domain)
    # Uklanjamo višestruke underscore-ove
    domain = re.sub(r'_+', '_', domain)
    # Uklanjamo leading/trailing underscore
    domain = domain.strip('_')
    return domain


def get_domain_folder(domain: str) -> str:
    """Generiše ime foldera za domain"""
    sanitized_domain = sanitize_domain_for_filename(domain)
    return os.path.join("storage", "mpanel", sanitized_domain)


def ensure_domain_folder_exists(domain: str) -> str:
    """Kreira folder za domain ako ne postoji i vraća putanju"""
    folder_path = get_domain_folder(domain)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def get_processed_ids_filename(domain: str) -> str:
    """Generiše ime fajla za obrađene ID-jeve na osnovu domain-a"""
    return "processed_ids.json"


def get_page_counter_filename(domain: str) -> str:
    """Generiše ime fajla za page brojač na osnovu domain-a"""
    return "page_counter.json"


def load_page_counter(domain: str) -> int:
    """Učitava trenutni page broj za domain"""
    folder_path = ensure_domain_folder_exists(domain)
    filename = get_page_counter_filename(domain)
    filepath = os.path.join(folder_path, filename)
    
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('current_page', 1)
        return 1
    except Exception as e:
        print(f"⚠️ Greška pri učitavanju page brojača: {e}")
        return 1


def save_page_counter(domain: str, page: int) -> None:
    """Čuva trenutni page broj za domain"""
    folder_path = ensure_domain_folder_exists(domain)
    filename = get_page_counter_filename(domain)
    filepath = os.path.join(folder_path, filename)
    
    try:
        data = {
            'current_page': page,
            'last_updated': datetime.now().isoformat(),
            'domain': domain
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"📄 Page brojač ažuriran: {page}")
        
    except Exception as e:
        print(f"❌ Greška pri čuvanju page brojača: {e}")


def increment_page_counter(domain: str) -> int:
    """Povećava page brojač za domain i čuva ga"""
    current_page = load_page_counter(domain)
    next_page = current_page + 1
    save_page_counter(domain, next_page)
    return next_page


def reset_page_counter(domain: str) -> None:
    """Resetuje page brojač na 1"""
    save_page_counter(domain, 1)
    print(f"🔄 Page brojač resetovan na 1 za domain: {domain}")


def show_page_counter(domain: str) -> None:
    """Prikazuje trenutni page brojač"""
    current_page = load_page_counter(domain)
    print(f"📄 Trenutni page za {domain}: {current_page}")


def load_processed_ids(domain: str) -> Set[int]:
    """Učitava ID-jeve obrađenih članaka iz JSON fajla"""
    folder_path = ensure_domain_folder_exists(domain)
    filename = get_processed_ids_filename(domain)
    filepath = os.path.join(folder_path, filename)
    
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data)
        return set()
    except Exception as e:
        print(f"⚠️ Greška pri učitavanju obrađenih ID-jeva: {e}")
        return set()


def save_processed_ids(processed_ids: Set[int], domain: str) -> None:
    """Čuva ID-jeve obrađenih članaka u JSON fajl u domain folder-u"""
    folder_path = ensure_domain_folder_exists(domain)
    filename = get_processed_ids_filename(domain)
    filepath = os.path.join(folder_path, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(list(processed_ids), f, indent=2, ensure_ascii=False)
            
        print(f"💾 Obrađeni ID-jevi sačuvani u: {folder_path}/{filename}")
        
    except Exception as e:
        print(f"❌ Greška pri čuvanju obrađenih ID-jeva: {e}")


def ensure_excel_folder_exists() -> str:
    """Kreira folder za Excel fajlove ako ne postoji"""
    excel_folder = os.path.join("storage", "excel")
    os.makedirs(excel_folder, exist_ok=True)
    return excel_folder


def get_excel_filepath() -> str:
    """Vraća putanju do Excel fajla"""
    excel_folder = ensure_excel_folder_exists()
    return os.path.join(excel_folder, "mpanel_persons.xlsx")


def load_existing_persons() -> pd.DataFrame:
    """Učitava postojeće osobe iz Excel fajla"""
    filepath = get_excel_filepath()
    
    try:
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            print(f"📊 Učitano {len(df)} postojećih osoba iz Excel-a")
            return df
        else:
            # Kreiraj novi DataFrame sa potrebnim kolonama
            df = pd.DataFrame(columns=['name', 'last_name', 'number_of_repeat'])
            print("📊 Kreiran novi Excel fajl")
            return df
    except Exception as e:
        print(f"⚠️ Greška pri učitavanju Excel-a: {e}")
        # Kreiraj novi DataFrame u slučaju greške
        return pd.DataFrame(columns=['name', 'last_name', 'number_of_repeat'])


def save_persons_to_excel(df: pd.DataFrame) -> None:
    """Čuva DataFrame u Excel fajl"""
    try:
        filepath = get_excel_filepath()
        df.to_excel(filepath, index=False)
        print(f"💾 Osobe sačuvane u: {filepath}")
    except Exception as e:
        print(f"❌ Greška pri čuvanju Excel-a: {e}")


def split_full_name(full_name: str) -> tuple:
    """Razdvaja puno ime na ime i prezime"""
    words = full_name.strip().split()
    if len(words) >= 2:
        # Prva reč je ime, sve ostale reči su prezime
        first_name = words[0]
        last_name = ' '.join(words[1:])
        return first_name, last_name
    else:
        return full_name, ""


def add_persons_to_excel(filtered_names: List[str]) -> None:
    """Dodaje nove osobe u Excel fajl ili povećava broj ponavljanja"""
    if not filtered_names:
        return
    
    # Učitaj postojeće osobe
    df = load_existing_persons()
    
    # Brojač novih i ažuriranih osoba
    new_persons = 0
    updated_persons = 0
    
    for full_name in filtered_names:
        first_name, last_name = split_full_name(full_name)
        
        # Proveri da li osoba već postoji
        existing_person = df[(df['name'] == first_name) & (df['last_name'] == last_name)]
        
        if existing_person.empty:
            # Dodaj novu osobu
            new_row = pd.DataFrame({
                'name': [first_name],
                'last_name': [last_name],
                'number_of_repeat': [1]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            new_persons += 1
            print(f"➕ Dodata nova osoba: {first_name} {last_name}")
        else:
            # Povećaj broj ponavljanja
            index = existing_person.index[0]
            df.at[index, 'number_of_repeat'] += 1
            updated_persons += 1
            print(f"🔄 Povećan broj ponavljanja za: {first_name} {last_name} (sada: {df.at[index, 'number_of_repeat']})")
    
    # Sačuvaj ažurirani Excel
    save_persons_to_excel(df)
    
    print(f"📊 Rezultat: {new_persons} novih osoba, {updated_persons} ažuriranih osoba")


def process_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesira pojedinačni članak
    
    Args:
        article (dict): Podaci o članku
        
    Returns:
        dict: Rezultat procesiranja
    """
    article_id = article.get('id')
    title = article.get('title', 'Bez naslova')
    content = article.get('contents', 'Bez sadržaja')
    
    
    # OVDE DODAJTE VAŠU LOGIKU ZA PROCESIRANJE ČLANKA
    # ================================================
    try:
        # Uvoz OpenAIService
        from app.services.openai_service import OpenAIService
                    
        # Inicijalizacija OpenAI servisa
        openai_service = OpenAIService()
        
        # Priprema parametara za API poziv
        schema = openai_service.get_human_names_schema()
        messages = [
            {
                "role": "system",
                "content": """You are a strict entity extraction assistant. Your task is to extract only **real, notable people** mentioned in the provided text.

                RULES:
                1. Only extract names of **real, famous individuals**.
                2. You must return only **full names**, meaning both **first name and last name** must be present.
                - Do NOT return partial names.
                - Do NOT return only a surname or only a given name.
                - Do NOT return initials (e.g., "S.T." or "Stefan T.").
                3. All names must be returned strictly in **nominative case** (the base grammatical form of the name, e.g., "Stefan Tomić", not "Stefanu Tomiću").
                4. If a person is mentioned multiple times in the text (including in different grammatical forms or spellings), include them **only once**, using their nominative form.
                5. Only include real people, not fictional characters, organizations, or other entities.

                OUTPUT FORMAT:
                - Return a JSON array of strings with full names in nominative form.
                - Example: ["Angela Merkel", "Elon Musk"]
                - If there are no valid full names, return an empty list: []

                EXAMPLES (valid):
                ✅ "Angela Merkel"
                ✅ "Elon Musk"
                ✅ "Stefan Tomić"

                EXAMPLES (invalid, must be excluded):
                ❌ "Merkel"
                ❌ "Stefan T."
                ❌ "S.T."
                ❌ "Stefanu Tomiću" (must be converted to "Stefan Tomić")
                ❌ "Ministar zdravlja"
                """
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": f"Text: {content}" }
                ]
            }
        ]
        
        # Poziv OpenAI API-ja
        response = openai_service.safe_openai_request(
            model="gpt-4.1",  # Koristite model koji je dostupan
            messages=messages,
            temperature=0.3,
            max_tokens=2100,
            functions=[schema],
            function_call={"name": "get_human_names"}
        )
        
        # Obrada odgovora
        if response.choices and response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            arguments = json.loads(function_call.arguments)
            print(f"Response: {json.dumps(arguments, ensure_ascii=False, indent=4)}")
            
            # Provera da li je osoba stvarna
            names = arguments.get('names', [])
            
            # Filtriranje imena - uklanjamo imena sa jednom rečju ili rečima sa manje od 3 slova
            filtered_names = []
            for name in names:
                # Razdvajamo ime na reči
                words = name.strip().split()
                
                # Proveravamo da li ima više od jedne reči
                if len(words) < 2:
                    print(f"❌ Uklanjam '{name}' - ima samo {len(words)} reč")
                    continue
                
                # Proveravamo da li svaka reč ima bar 3 slova
                valid_words = True
                for word in words:
                    if len(word) < 3:
                        print(f"❌ Uklanjam '{name}' - reč '{word}' ima manje od 3 slova")
                        valid_words = False
                        break
                
                if valid_words:
                    filtered_names.append(name)
                    print(f"✅ Zadržavam '{name}' - validno ime")
            
            if filtered_names:
                print(f"✅ Konačne stvarne osobe: {filtered_names}")
                # Čuvamo osobe u Excel fajl
                add_persons_to_excel(filtered_names)
            else:
                print(f"❌ Nisu pronađene validne stvarne osobe u članku.")

            
    except ImportError as e:
        print(f"❌ Greška pri import-u modula: {str(e)}")
        print("Proverite da li je app modul dostupan")
    except Exception as e:
        print(f"❌ Greška prilikom obrade članka: {str(e)}")
    # Primer procesiranja - možete prilagoditi prema vašim potrebama
    processed_data = {
        'id': article_id,
        'title': title,
        'intro': article.get('intro'),
        'publish_date': article.get('publish_date'),
        'url': article.get('url'),
        'categories': [cat.get('name') for cat in article.get('categories', [])],
        'tags': [tag.get('name') for tag in article.get('tags', [])],
        'featured_image': article.get('featured_image'),
        'processed_at': datetime.now().isoformat(),
        'status': 'processed'
    }
    
    # Dodatna obrada - možete dodati svoju logiku ovde
    # processed_data['custom_field'] = your_custom_processing(article)
    
    # ================================================
    
    return processed_data


def process_articles_from_response(api_response: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Procesira sve članke iz API odgovora
    
    Args:
        api_response (dict): Odgovor od API-ja
        domain (str): Domain URL za generisanje imena fajla
        
    Returns:
        dict: Statistika procesiranja
    """
    if api_response.get('status') != 'success':
        print("❌ API odgovor nije uspešan, preskačem procesiranje")
        return {'error': 'API response not successful'}
    
    # Učitavamo obrađene ID-jeve
    processed_ids = load_processed_ids(domain)
    
    # Izvlačimo članke iz odgovora
    data = api_response.get('data', {})
    result = data.get('result', {})
    articles = result.get('articles', [])
    
    if not articles:
        print("ℹ️ Nema članaka za procesiranje")
        return {'processed': 0, 'skipped': 0, 'total': 0}
    
    print(f"📚 Pronađeno {len(articles)} članaka")
    print(f"📊 Trenutno obrađeno članaka: {len(processed_ids)}")
    
    processed_count = 0
    skipped_count = 0
    
    for article in articles:
        article_id = article.get('id')
        
        if not article_id:
            print("⚠️ Članak bez ID-ja, preskačem")
            continue
        
        if article_id in processed_ids:
            print(f"⏭️ Članak {article_id} već obrađen, preskačem")
            skipped_count += 1
            continue
        
        # Procesiranje članka
        try:
            processed_article = process_article(article)
            
            # Dodajemo ID u listu obrađenih
            processed_ids.add(article_id)
            processed_count += 1
            
            # OVDE MOŽETE DODATI ČUVANJE REZULTATA PROCESIRANJA
            # ==================================================
            # Primer: Čuvanje u zaseban fajl
            # with open(f"article_{article_id}.json", 'w', encoding='utf-8') as f:
            #     json.dump(processed_article, f, indent=2, ensure_ascii=False)
            # ==================================================
            
        except Exception as e:
            print(f"❌ Greška pri procesiranju članka {article_id}: {e}")
    
    # Čuvamo ažuriranu listu obrađenih ID-jeva
    save_processed_ids(processed_ids, domain)
    
    return {
        'processed': processed_count,
        'skipped': skipped_count,
        'total': len(articles),
        'total_processed_ids': len(processed_ids)
    }


def call_api_endpoint(domain: str, header_value: str, article_limit: int = 25, page: int = 1, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Poziva GET API endpoint na prosleđenom domain-u
    
    Args:
        domain (str): Domain URL (npr. 'https://api.example.com')
        header_value (str): Vrednost za header
        article_limit (int): Broj artikala (default: 25)
        page (int): Broj stranice (default: 1)
        data (dict): Podaci za slanje (opciono)
        
    Returns:
        dict: Odgovor od API-ja
    """
    # Uklanjamo trailing slash iz domain-a
    domain = domain.rstrip('/')
    
    # Fiksni endpoint sa query parametrima
    endpoint = f"/api/webV2/getArticles?articleLimit={article_limit}&page={page}"
    url = f"{domain}{endpoint}"
    
    # Default podaci ako nisu prosleđeni
    if data is None:
        data = {
            "message": "Test request",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    # Headers sa prosleđenim header-om
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'FaceRecognitionWeb-APIClient/1.0',
        'Authorization': header_value  # Dodajemo prosleđeni header
    }
    
    try:
        print(f"🌐 Pozivanje: {url}")
        print(f"🔑 Header: Authorization: {header_value}")
        
        # Izvršavanje GET zahteva
        response = requests.get(url, json=data, headers=headers, timeout=30)
        
        # Proveravamo status kod
        response.raise_for_status()
        
        # Pokušavamo da parsijemo JSON odgovor
        try:
            result = response.json()
            return {
                'status': 'success',
                'status_code': response.status_code,
                'data': result,
                'message': 'Uspešno pozvan API endpoint'
            }
        except json.JSONDecodeError:
            return {
                'status': 'success',
                'status_code': response.status_code,
                'data': response.text,
                'message': 'Odgovor nije JSON format'
            }
            
    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'error': str(e),
            'message': f'Greška pri komunikaciji sa API-jem: {e}'
        }


def display_response(response: Dict[str, Any]) -> None:
    """
    Prikazuje odgovor od API-ja
    
    Args:
        response (dict): Odgovor od API-ja
    """
    print("\n" + "="*50)
    print("ODGOVOR OD API-JA")
    print("="*50)
    
    if response.get('status') == 'error':
        print(f"❌ GREŠKA: {response.get('message', 'Nepoznata greška')}")
        if 'error' in response:
            print(f"Detalji: {response['error']}")
    else:
        print("✅ USPEŠNO")
        print(f"Status kod: {response.get('status_code', 'N/A')}")
        print(f"Poruka: {response.get('message', 'N/A')}")
        
    
    print("="*50 + "\n")


def main():
    """Glavna funkcija skripte"""
    parser = argparse.ArgumentParser(description='API GET Client sa procesiranjem članaka')
    parser.add_argument('domain', help='Domain URL (npr. https://api.example.com)')
    parser.add_argument('--header', required=True, help='Vrednost za Authorization header')
    parser.add_argument('--article-limit', type=int, default=25, help='Broj artikala (default: 25)')
    parser.add_argument('--page', type=int, help='Broj stranice (opciono - ako nije prosleđen, koristi se automatski)')
    parser.add_argument('--data', help='JSON podaci za slanje (opciono)')
    parser.add_argument('--reset-page', action='store_true', help='Resetuje page brojač na 1')
    parser.add_argument('--show-page', action='store_true', help='Prikazuje trenutni page brojač')
    
    args = parser.parse_args()
    
    # Specijalne komande
    if args.reset_page:
        reset_page_counter(args.domain)
        return
    
    if args.show_page:
        show_page_counter(args.domain)
        return
    
    # Parsiranje JSON podataka ako su prosleđeni
    data = None
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(f"❌ Greška pri parsiranju JSON podataka: {e}")
            sys.exit(1)
    
    # Određivanje page broja
    if args.page is not None:
        # Ako je prosleđen page, koristimo ga
        page = args.page
        print(f"📄 Korišćen prosleđeni page: {page}")
    else:
        # Ako nije prosleđen, učitavamo automatski page
        page = load_page_counter(args.domain)
        print(f"📄 Automatski učitavan page: {page}")
    
    # Pozivanje API endpoint-a
    response = call_api_endpoint(args.domain, args.header, args.article_limit, page, data)
    
    # Prikazivanje odgovora
    display_response(response)
    
    # Procesiranje članaka
    print("\n" + "="*50)
    print("PROCESIRANJE ČLANAKA")
    print("="*50)
    
    stats = process_articles_from_response(response, args.domain)
    
    # Prikazivanje statistike
    print("\n" + "="*50)
    print("STATISTIKA PROCESIRANJA")
    print("="*50)
    
    if 'error' in stats:
        print(f"❌ Greška: {stats['error']}")
    else:
        print(f"✅ Obrađeno: {stats['processed']}")
        print(f"⏭️ Preskočeno: {stats['skipped']}")
        print(f"📚 Ukupno u odgovoru: {stats['total']}")
        print(f"📊 Ukupno obrađeno članaka: {stats['total_processed_ids']}")
        
        # Povećavamo page brojač ako:
        # 1. Bilo je obrađenih članaka, ILI
        # 2. Svi članci su preskočeni (već obrađeni) - to znači da smo završili sa ovim page-om
        if stats['processed'] > 0 or (stats['skipped'] > 0 and stats['processed'] == 0):
            next_page = increment_page_counter(args.domain)
            print(f"📄 Page brojač povećan na: {next_page}")
            
            # Ako su svi članci preskočeni, obavestimo korisnika
            if stats['skipped'] > 0 and stats['processed'] == 0:
                print(f"ℹ️ Svi članci na stranici {page} su već obrađeni, prelazim na sledeću stranicu")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    main() 