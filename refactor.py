import requests
import json

# Your working API setup
url = "https://displaycatalog.mp.microsoft.com/v7.0/products"
# Just use one or two IDs to get a clean sample
product_ids = ["9N7271QN4SGB"] # Mortal Kombat 1
params = {"bigIds": ",".join(product_ids), "market": "US", "languages": "en-US"}
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            all_tidy_results = []
            # Corrected: Iterate through every product in the API response
            for p in data.get("Products", []):
                lp = p.get("LocalizedProperties", [{}])[0]
                mp = p.get("MarketProperties", [{}])[0]
                prop = p.get("Properties", {})
                usage = mp.get("UsageData", [])
                ratings = mp.get("ContentRatings", [])

                tidy = {
                    "product_id": p.get("ProductId"),
                    "title": lp.get("ProductTitle"),
                    "publisher": lp.get("PublisherName"),
                    "developer": lp.get("DeveloperName"),
                    "release_date": mp.get("OriginalReleaseDate"),
                    "short_description": lp.get("ShortDescription"),
                    "rating_all_time": (mp.get("UsageData") or [])[-1] if mp.get("UsageData") else None,
                    "rating_7_days": (mp.get("UsageData") or [])[-3] if mp.get("UsageData") and len(mp.get("UsageData")) > 1 else None,
                    "rating_30_days": (mp.get("UsageData") or [])[-2] if mp.get("UsageData") and len(mp.get("UsageData")) > 2 else None,
                    "bundle_count": len(p.get("Properties", {}).get("BundledSkus", [])),
                    "is_xpa": p.get("Properties", {}).get("XboxXPA", False),
                    "platforms": p.get("Properties", {}).get("SupportedPlatforms", []),
                    "asset_count": len(lp.get("Images", [])) + len(lp.get("Videos", [])) + len(lp.get("CMSVideos", [])),
                    "has_gamepass_remediation": any(
                        "Game Pass" in rem.get("Description", "") 
                        for lp_item in p.get("LocalizedProperties", [])
                        for rem in lp_item.get("EligibilityProperties", {}).get("Remediations", [])
                    ),
                    "category": prop.get("Category", []),
                    "esrb_descriptors": [
                        desc.split(":")[-1] 
                        for r in ratings if r.get("RatingSystem") == "ESRB"
                        for desc in r.get("RatingDescriptors", [])
                          ],
                    "prices": [
                        {
                            "list_price": a.get("OrderManagementData", {}).get("Price", {}).get("ListPrice"),
                            "msrp": a.get("OrderManagementData", {}).get("Price", {}).get("MSRP"),
                            "start": a.get("Conditions", {}).get("StartDate"),
                            "end": a.get("Conditions", {}).get("EndDate"),
                        }
                        for sku in p.get("DisplaySkuAvailabilities", [])
                        for a in sku.get("Availabilities", [])
                        if a.get("OrderManagementData", {}).get("Price")
                    ]
                }
                all_tidy_results.append(tidy)
        # SAVE IT LOCALL
        with open("tidy_results.json", "w") as f:
            json.dump(all_tidy_results, f, indent=4)
        print("Success! Cleaned data saved to tidy_results.json")

        # 2. Save the RAW API response - Good for unit testing later
        with open("real_api_raw_sample.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Success! Raw API data saved to real_api_raw_sample.json")
except requests.RequestException as e:
    print(f"Error fetching data from API: {e}")