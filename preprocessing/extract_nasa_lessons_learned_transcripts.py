import yt_dlp
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import re
import os
from bs4 import BeautifulSoup
import requests

import glob

def setup_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    performance_log_prefs = {'goog:loggingPrefs': {'performance': 'ALL'}}
    chrome_options.set_capability('goog:loggingPrefs', performance_log_prefs['goog:loggingPrefs'])
    return webdriver.Chrome(options=chrome_options)

def get_mp4_link(video_id):
    
    driver = setup_chrome_driver()
    try:
    
        print(f"Navigating to https://mediaex-server.larc.nasa.gov/Academy/Play/{video_id}")
        driver.get(f"https://mediaex-server.larc.nasa.gov/Academy/Play/{video_id}")
        time.sleep(3)
        video_element = driver.find_element(By.TAG_NAME, 'video')
        ActionChains(driver).move_to_element(video_element).click().perform()
        time.sleep(3)
        logs = driver.get_log("performance")
        regex = r"https://media-iis\.larc\.nasa\.gov/MediasiteDeliver/MP4/([a-f0-9\-]+)\.mp4"
        for log in logs:
            match = re.search(regex, log["message"])
            if match:
                print("Extracted video link id: ", match.group(1))
                return f"https://media-iis.larc.nasa.gov/MediasiteDeliver/MP4/{match.group(1)}.mp4/QualityLevels(338000)"
        return None
    
    finally:
        driver.quit()

def download_audio(video_url, output_filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def setup_transcription_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

def transcribe_audio(pipe, audio_file):
    print(f"Audio file: {audio_file}")
    return pipe(audio_file)["text"]

def process_video(video_info, transcription_pipeline, output_path):
    
    print(f"Processing video: {video_info['title']}")

    # Search for files containing video_info['id'] in their name within the specified folder
    ## check if file exists before downloading
    existing_files = glob.glob(f"{output_path}/audio/*{video_info['id']}*.mp3")

    audio_filename = f"{output_path}/audio/{video_info['id']}_audio"
    if existing_files:
        print(f"Audio already exists for video {video_info['title'].replace(' ','_')} with ID: {video_info['id']}")
    else:
        mp4_link = get_mp4_link(video_info['id'])
        if not mp4_link:
            print(f"Failed to get MP4 link for video ID: {video_info['id']}")
            return
        download_audio(mp4_link, audio_filename)
    
    existing_files = glob.glob(f"{output_path}/transcript/*{video_info['id']}*.txt")
    
    if existing_files:
        print(f"Transcript already exists for video {video_info['title'].replace(' ','_')} with ID: {video_info['id']}")
    
    else:
        
        transcript = transcribe_audio(transcription_pipeline, audio_filename+".mp3")
        output_filename = f"{output_path}/transcript/{video_info['id']}_transcript.txt"
        with open(output_filename, "w") as f:
            f.write(f"Title: {video_info['title']}\n")
            f.write(f"Presenter: {video_info['presenter']}\n")
            f.write(f"Date: {video_info['date']}\n")
            #f.write(f"Category: {video_info['category']}\n\n")
            f.write(transcript)
    
    # Sanitize the title by replacing invalid characters with a safe character (e.g., space)
    safe_title = re.sub(r'[<>:"/\\|?*]', '', video_info['title']).strip()
    safe_title = safe_title.replace(" ", "_")

    print(f"Sanitized title: {safe_title}")

    # rename transcript files after processing
    if os.path.exists(f"{output_path}/transcript/{video_info['id']}_transcript.txt"):
        os.rename(f"{output_path}/transcript/{video_info['id']}_transcript.txt", f"{output_path}/transcript/{safe_title}_{video_info['id']}.txt")

    print(f"Completed processing for video: {video_info['title']}")
    #os.remove(audio_filename)  # Clean up the audio file

def extract_video_info(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    video_cards = soup.find_all('div', class_='card video-card bg-transparent border-dark swiper-slide me-5')
    
    # Filter the divs to only include those with data-categories=".Lessons Learned"
    video_cards = [div for div in video_cards if div.get('data-categories') == '.Lessons Learned']
    print(f"Found {len(video_cards)} videos with the category 'Lessons Learned'")
    
    video_info_list = []
    for card in video_cards:
        video_info = {
            'id': card.find('a')['href'].split('/')[-1],
            'title': card.find('h5', class_='card-image-title').text.strip(),
            'presenter': card.find('p', class_='card-text video-presenter').text.strip(),
            'date': card.find('p', class_='card-text video-record_date').text.strip(),
        }
        video_info_list.append(video_info)
    print([id['id'] for id in video_info_list])
    
    return video_info_list

def main():
    # HTML content would typically be fetched from a file or a web request
    # For this example, we'll use a placeholder string

    url = "https://nescacademy.nasa.gov/catalogs/spacesuit"    
        
    # Make the request to fetch the content
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        html_content = response.text  # Get the HTML content as text
        #print(html_content)  # Optional: Print the HTML content to verify
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")


    video_info_list = extract_video_info(html_content)
    print(video_info_list)
    
    transcription_pipeline = setup_transcription_pipeline()
    
    for video_info in video_info_list:
        process_video(video_info, transcription_pipeline, "../datasets/nasa_teaching_transcripts")


if __name__ == "__main__":
    main()