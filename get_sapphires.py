from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import progressbar
from tenacity import retry, stop_after_attempt, wait_fixed

import requests
import re
import csv

# Extract URLs and names of sapphires
def get_sapphire_urls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    product_divs = soup.find_all('div', class_='product_Container')

    products = []
    for product in product_divs:
        # Extract the product name
        name_tag = product.find('strong', itemprop='name')
        name = name_tag.text.strip() if name_tag else None

        # Extract the product URL
        url_tag = product.find('a', itemprop='url')
        url = url_tag['href'] if url_tag else None

        if name and url:
            products.append({"name": name, "url": url})

    return products

# Extract sapphire details from the product page
def parse_sapphire_details(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    specs_table = soup.find('table', class_='specs')
    sapphire_details = {}

    first_asset_li = soup.find('li', class_='Asset')
    if first_asset_li:
        # Find the <img> tag with class 'img-responsive' within the <li>
        img_tag = first_asset_li.find('img', class_='img-responsive')
        if img_tag and 'src' in img_tag.attrs:
            # Return the image URL (src attribute)
            sapphire_details['image_url'] = img_tag['src']


    if specs_table:
        rows = specs_table.find_all('tr')
        for row in rows:
            # Extract the header (key) and the corresponding data (value)
            key_cell = row.find('th')
            value_cell = row.find('td')

            if key_cell and value_cell:
                # Clean up the text to remove extra whitespace and tooltips
                key = key_cell.get_text(strip=True).replace(':', '')  # Remove trailing colon
                value = value_cell.get_text(strip=True)
                sapphire_details[key] = value

    return sapphire_details

# Update a single sapphire's details
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))  # Retry 5 times, wait 2 seconds between attempts
def update_one_sapphire_url(sapphire):
    url = 'https://www.thenaturalsapphirecompany.com' + sapphire['url']
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    sapphire_html = response.text
    sapphire_details = parse_sapphire_details(sapphire_html)
    sapphire.update(sapphire_details)

# Main script
def refresh_sapphire_csv():
    main_page_urls = [
        'https://www.thenaturalsapphirecompany.com/padparadscha-sapphires/?pagesize=1000',
        # 'https://www.thenaturalsapphirecompany.com/white-sapphires/?pagesize=1000',
        # THis one is just peach, comment out if doing all uniques
        'https://www.thenaturalsapphirecompany.com/unique-colored-sapphires/?color=peach&price_min=0&price_max=350000&carat_min=0&carat_max=35&sortby=&pagesize=1000',
        # 'https://www.thenaturalsapphirecompany.com/unique-colored-sapphires/?pagesize=5000',
        # 'https://www.thenaturalsapphirecompany.com/unique-colored-sapphires/?pagenum=2&pagesize=5000',
        # 'https://www.thenaturalsapphirecompany.com/blue-sapphires/?pagesize=5000',
        # 'https://www.thenaturalsapphirecompany.com/blue-sapphires/?pagenum=2&pagesize=5000',
        # 'https://www.thenaturalsapphirecompany.com/pink-sapphires/?pagesize=3000',
        # 'https://www.thenaturalsapphirecompany.com/yellow-sapphires/?pagesize=3000',
        # 'https://www.thenaturalsapphirecompany.com/green-sapphires/?pagesize=1000',
        # 'https://www.thenaturalsapphirecompany.com/purple-sapphires/?pagesize=1000',
    ]
    sapphires = []
    for base_url in main_page_urls:
        parsed_html = requests.get(base_url).text

        # Step 1: Get sapphire URLs
        sapphire_urls = get_sapphire_urls(parsed_html)
        sapphires.extend(sapphire_urls)
        print(f"Found {len(sapphire_urls)} sapphires on {base_url}")

    # Step 2: Update sapphire details using multithreading
    bar = progressbar.ProgressBar(max_value=len(sapphires))
    with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust max_workers based on system capacity
        futures = {executor.submit(update_one_sapphire_url, sapphire): sapphire for sapphire in sapphires}

        # Optional: Progress tracking
        for future in as_completed(futures):
            sapphire = futures[future]
            try:
                future.result()  # Wait for thread to finish and handle exceptions
            except Exception as exc:
                sapphire['error'] = str(exc)
            bar.update(bar.value + 1)
    bar.finish()

    # Cleanup
    for sapphire in sapphires:
        sapphire['Total Price'] = sapphire['Total Price'].replace('$', '').replace(',', '').strip() if 'Total Price' in sapphire else None
        sapphire['Weight'] = sapphire['Weight'].replace(' Ct.', '').strip() if 'Weight' in sapphire else None
        lwh = sapphire['Dimensions (mm)']
        lwh = re.sub(r'\s+', ' ', lwh).strip()
        match = re.match(r'([\d.]+)L x ([\d.]+)W x ([\d.]+)H', lwh)
        length, width, height = map(float, match.groups())
        sapphire['Length'] = length
        sapphire['Width'] = width
        sapphire['Height'] = height
        del sapphire['Dimensions (mm)']
        sapphire['url'] = 'https://www.thenaturalsapphirecompany.com' + sapphire['url']
        sapphire['Price per Length'] = float(sapphire['Total Price']) / length if length > 0 else None

    # Step 3: Save sapphire details to a CSV file
    with open('sapphires.csv', 'w', newline='') as csvfile:
        fieldnames = sapphires[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sapphires)

if __name__ == "__main__":
    refresh_sapphire_csv()