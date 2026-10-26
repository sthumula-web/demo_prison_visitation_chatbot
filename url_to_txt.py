import requests

from bs4 import BeautifulSoup

ur1 = "https://www.pa.gov/agencies/cor/frequently-asked-questions"
response = requests.get(ur1, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.text, "html.parser")

# Clean up the text - remove excessive blank lines
text = soup.get_text()
lines = [line.strip() for line in text.splitlines() if line.strip()]
clean_text = "\n".join(lines)

# Save to a file in your demo folder
with open("pa_doc_faqs.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("Done! FAQ text saved to pa_doc_faqs.txt")
print(f"Total characters saved: {len(clean_text)}")



