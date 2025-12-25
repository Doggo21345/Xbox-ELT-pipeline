import azure.functions as func
import logging
import requests
import json
import pandas as pd
from datetime import datetime
import time
from azure.storage.blob import BlobServiceClient
import os

app = func.FunctionApp()

@app.timer_trigger(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=False, use_monitor=False) 
def Xbox_Time_Trigger_GP_MS(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Starting Xbox Store Scrape...')

    # 1. LOAD YOUR CLEANED IDs
    # Ensure this CSV is in your 'scraper-service' folder
    try:
        df = pd.read_csv('xbox_final_cleaned_results.csv')
        product_ids = df['ProductID'].dropna().unique().tolist()
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return

    batch_size = 20
    all_tidy_results = []
    
    url = "https://displaycatalog.mp.microsoft.com/v7.0/products"

    for i in range(0, len(product_ids), batch_size):
        batch = product_ids[i:i + batch_size]
        batch_string = ",".join(batch)
        logging.info(f"Scraping batch {i//batch_size + 1}...")
        
        params = {"bigIds": batch_string, "market": "US", "languages": "en-US"}
        
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                # Corrected: Iterate through every product in the API response
                for p in data.get("Products", []):
                    lp = p.get("LocalizedProperties", [{}])[0]
                    mp = p.get("MarketProperties", [{}])[0]

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
            
            # Rate limit safety
            time.sleep(2) 
            
        except Exception as e:
            logging.error(f"Error in batch starting at index {i}: {e}")

    # 4. SAVE TO AZURE BLOB STORAGE
    if all_tidy_results:
        save_to_azure_blob(all_tidy_results)

def save_to_azure_blob(results):
    # AzureWebJobsStorage is the default env var for the linked storage account
    connect_str = os.getenv('AzureWebJobsStorage:') 
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    # Ensure the container name matches what you create in the portal
    container_name = "xbox-data"
    filename = f"scrapes/xbox_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    blob_client.upload_blob(json.dumps(results, indent=2), overwrite=True)
    logging.info(f"Successfully uploaded {filename} to Azure!")