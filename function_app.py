import azure.functions as func
import logging
import requests
import json
import pandas as pd
from datetime import datetime
import time
from azure.storage.blob import BlobServiceClient
import os
import streamlit as st

app = func.FunctionApp()

@app.timer_trigger(schedule="0 0 0 1/14 * *", arg_name="myTimer", run_on_startup=False, use_monitor=False) 
def Xbox_Time_Trigger_GP_MS(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Starting Xbox Store Scrape...')

    try:
        script_dir = os.path.dirname(__file__)
        csv_path = os.path.join(script_dir, 'xbox_final_cleaned_results.csv')
        df = pd.read_csv(csv_path)
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
                
                for p in data.get("Products", []):
                    # Helper variables
                    localized_props = p.get("LocalizedProperties", [])
                    lp = localized_props[0] if localized_props else {}
                    
                    market_props = p.get("MarketProperties", [])
                    mp = market_props[0] if market_props else {}
                    
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
                        "rating_alltime": usage[-1] if usage else None,
                        "rating_7_days": usage[-3] if len(usage) > 2 else None,
                        "rating_30_days": usage[-2] if len(usage) > 1 else None,
                        "bundle_count": len(prop.get("BundledSkus", [])),
                        "is_xpa": prop.get("XboxXPA", False),
                        "asset_count": len(lp.get("Images", [])) + len(lp.get("Videos", [])) + len(lp.get("CMSVideos", [])),
                        "category": prop.get("Category", "unknown"),
                        "esrb": next(
                            (r.get("RatingId", "").split(":")[-1] 
                             for r in ratings if r.get("RatingSystem") == "ESRB"), 
                            None
                        ),
                        "esrb_descriptors": [
                        desc.split(":")[-1] 
                        for r in ratings if r.get("RatingSystem") == "ESRB"
                        for desc in r.get("RatingDescriptors", [])
                          ],
                        "has_gamepass_remediation": any(
                            "Game Pass" in rem.get("Description", "") 
                            for lp_item in localized_props
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
            
            time.sleep(1) # Rate limit safety
            
        except Exception as e:
            logging.error(f"Error in batch starting at index {i}: {e}")

    if all_tidy_results:
        save_to_azure_blob(all_tidy_results)

def save_to_azure_blob(results):
    connect_str = os.getenv("AzureWebJobsStorage")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    container_name = "xbox-data"
    filename = f"scrapes/xbox_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    blob_client.upload_blob(json.dumps(results, indent=2), overwrite=True)
    logging.info(f"Successfully uploaded {filename} to Azure!")
    
    # RE-ADDED: Call Databricks after upload
    trigger_databricks_job()
    

def trigger_databricks_job():
    server_hostname = os.environ.get("DATABRICKS_HOSTNAME")
    token = os.environ.get("DATABRICKS_TOKEN")
    job_id = os.environ.get("DATABRICKS_JOB_ID")

    workspace_url = f"https://{server_hostname}"
    endpoint = f"{workspace_url}/api/2.2/jobs/run-now"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"job_id": job_id}

    response = requests.post(endpoint, headers=headers, json=payload)
    
    if response.status_code == 200:
        logging.info(f"Databricks Job {job_id} triggered successfully!")
    else:
        logging.error(f"Failed to trigger Databricks: {response.text}")