import urllib.request
import xml.etree.ElementTree as ET
import os
import ssl

# SSL Certificate error එක bypass කිරීම
ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs('data', exist_ok=True)
url = 'https://export.arxiv.org/api/query?search_query=all:SMS+spam+detection&start=0&max_results=20'

print("📥 Downloading 20 research papers into data/ folder...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
xml_data = urllib.request.urlopen(req).read()
root = ET.fromstring(xml_data)

links = root.findall('{http://www.w3.org/2005/Atom}entry/{http://www.w3.org/2005/Atom}link[@title="pdf"]')

for i, link in enumerate(links, 1):
    pdf_url = link.attrib['href'].replace('abs', 'pdf') + '.pdf'
    pdf_req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(pdf_req) as response, open(f'data/paper_{i}.pdf', 'wb') as out_file:
        out_file.write(response.read())
    print(f"✅ Downloaded Paper {i}/20")

print("\n🎉 All 20 Papers Downloaded Successfully into data/ folder!")