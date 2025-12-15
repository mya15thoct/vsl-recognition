"""
QIPEDC Vietnamese Sign Language Dataset Preparation
Complete pipeline: Crawl -> Parse -> Download -> Extract Frames -> Create Dictionary
"""

import os
import json
import time
import re
import cv2
import requests
from pathlib import Path
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class QIPEDCDatasetPreparer:
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, 'qipedc_raw')
        self.dataset_dir = os.path.join(output_dir, 'VSL_Isolated')
        self.videos_dir = os.path.join(self.dataset_dir, 'videos')
        self.frames_dir = os.path.join(self.dataset_dir, 'frames')
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.driver = None
    
    # ==================== STEP 1: CRAWL ====================
    
    def crawl_qipedc(self):
        """Crawl data from QIPEDC website"""
        print("="*60)
        print("STEP 1: CRAWLING DATA FROM QIPEDC")
        print("="*60)
        
        print("\nInitializing Chrome driver...")
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        print(" Chrome initialized")
        
        base_url = 'https://qipedc.moet.gov.vn'
        
        # Load dictionary page
        print(f"\nLoading: {base_url}/dictionary")
        self.driver.get(f'{base_url}/dictionary')
        time.sleep(3)
        
        # Set to 80 words per page
        try:
            script = """
            document.getElementById('group').value = '80';
            document.getElementById('group').dispatchEvent(new Event('change'));
            """
            self.driver.execute_script(script)
            time.sleep(3)
            print("[OK] Set to 80 words per page")
        except:
            print("[WARNING] Could not change page size, using default")
        
        # Get total number of pages
        print("\nDetecting total pages...")
        try:
            pagination_script = """
            var pagination = document.querySelector('.pagination');
            if (pagination) {
                var pageLinks = pagination.querySelectorAll('a');
                var maxPage = 1;
                pageLinks.forEach(function(link) {
                    var pageNum = parseInt(link.textContent);
                    if (!isNaN(pageNum) && pageNum > maxPage) {
                        maxPage = pageNum;
                    }
                });
                return maxPage;
            }
            return 1;
            """
            total_pages = self.driver.execute_script(pagination_script)
            print(f"[OK] Found {total_pages} pages")
        except:
            total_pages = 1
            print("[WARNING] Could not detect pages, assuming 1 page")
        
        # Extract words from all pages
        all_words = []
        
        extract_script = """
        var words = [];
        var items = document.querySelectorAll('#product > a');
        items.forEach(function(item) {
            var text = item.querySelector('p') ? item.querySelector('p').innerText : '';
            var img = item.querySelector('img') ? item.querySelector('img').src : '';
            var onclick = item.getAttribute('onclick') || '';
            words.push({
                text: text,
                image: img,
                onclick: onclick
            });
        });
        return words;
        """
        
        for page_num in range(1, total_pages + 1):
            print(f"\nExtracting words from page {page_num}/{total_pages}...")
            
            # Navigate to specific page if not the first one
            if page_num > 1:
                try:
                    # Click on page number
                    page_link_script = f"""
                    var pageLinks = document.querySelectorAll('.pagination a');
                    for (var i = 0; i < pageLinks.length; i++) {{
                        if (pageLinks[i].textContent == '{page_num}') {{
                            pageLinks[i].click();
                            break;
                        }}
                    }}
                    """
                    self.driver.execute_script(page_link_script)
                    time.sleep(3)  # Wait for page to load
                except Exception as e:
                    print(f"  ⚠ Error navigating to page {page_num}: {e}")
                    continue
            
            # Extract words from current page
            try:
                page_words = self.driver.execute_script(extract_script)
                if page_words:
                    all_words.extend(page_words)
                    print(f"  [OK] Extracted {len(page_words)} words from page {page_num}")
                else:
                    print(f"  [WARNING] No words found on page {page_num}")
            except Exception as e:
                print(f"  [ERROR] Error extracting from page {page_num}: {e}")
        
        # Save all collected words
        if all_words:
            output_file = os.path.join(self.raw_dir, 'all_words_complete.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_words, f, ensure_ascii=False, indent=2)
            
            print(f"\n{'='*60}")
            print(f"[OK] Total extracted: {len(all_words)} words from {total_pages} pages")
            print(f"  Saved to: {output_file}")
            print(f"{'='*60}")
        
        self.driver.quit()
        return all_words
    
    # ==================== STEP 2: PARSE ====================
    
    def parse_onclick(self, onclick_str):
        """Parse onclick attribute to extract video info"""
        pattern = r"modalData\('([^']+)','([^']*)','([^']*)','([^']*)'?\s*\)"
        match = re.search(pattern, onclick_str)
        
        if match:
            return {
                'video_id': match.group(1),
                'word': match.group(2),
                'description': match.group(3),
                'has_video': match.group(4) == 'true'
            }
        return None
    
    def parse_data(self, words_data):
        """Parse crawled data and extract video URLs"""
        print("\n" + "="*60)
        print("STEP 2: PARSING DATA")
        print("="*60)
        
        parsed_words = []
        
        for item in words_data:
            onclick = item.get('onclick', '')
            if onclick:
                parsed = self.parse_onclick(onclick)
                if parsed:
                    word_entry = {
                        'video_id': parsed['video_id'],
                        'word': parsed['word'],
                        'description': parsed['description'],
                        'has_video': parsed['has_video'],
                        'image_url': item.get('image', ''),
                        'video_url': f"https://qipedc.moet.gov.vn/videos/{parsed['video_id']}.mp4"
                    }
                    parsed_words.append(word_entry)
        
        # Save parsed data
        output_file = os.path.join(self.raw_dir, 'parsed_words.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_words, f, ensure_ascii=False, indent=2)
        
        print(f"\n[OK] Parsed {len(parsed_words)} words")
        print(f"  Unique words: {len(set(w['word'] for w in parsed_words))}")
        print(f"  Saved to: {output_file}")
        
        return parsed_words
    
    # ==================== STEP 3: DOWNLOAD & EXTRACT ====================
    
    def download_video(self, url, output_path, max_retries=3):
        """Download video from URL"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        with tqdm(total=total_size, unit='B', unit_scale=True, leave=False) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"  [ERROR] Failed: {e}")
                    return False
    
    def extract_frames(self, video_path, output_folder):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            os.makedirs(output_folder, exist_ok=True)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_filename = os.path.join(output_folder, f'{frame_count:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
            
            cap.release()
            return True
            
        except Exception as e:
            print(f"  [ERROR] Extract error: {e}")
            return False
    
    def download_and_process(self, parsed_words):
        """Download videos and extract frames"""
        print("\n" + "="*60)
        print("STEP 3: DOWNLOADING VIDEOS & EXTRACTING FRAMES")
        print("="*60)
        
        successful = 0
        failed = []
        
        for idx, word_info in enumerate(parsed_words):
            print(f"\n[{idx+1}/{len(parsed_words)}] {word_info['word']}")
            
            # Setup paths
            class_id = f"{idx:06d}"
            video_path = os.path.join(self.videos_dir, f"{class_id}.mp4")
            signer_folder = os.path.join(self.frames_dir, class_id, '01')
            
            # Download video
            if not os.path.exists(video_path):
                print(f"  Downloading video...")
                success = self.download_video(word_info['video_url'], video_path)
                if not success:
                    failed.append(word_info)
                    continue
            else:
                print(f"  [OK] Video exists")
            
            # Extract frames
            if not os.path.exists(signer_folder) or len(os.listdir(signer_folder)) == 0:
                print(f"  Extracting frames...")
                success = self.extract_frames(video_path, signer_folder)
                if success:
                    num_frames = len([f for f in os.listdir(signer_folder) if f.endswith('.jpg')])
                    print(f"  [OK] Extracted {num_frames} frames")
                    successful += 1
            else:
                print(f"  [OK] Frames exist")
                successful += 1
        
        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  Total: {len(parsed_words)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(failed)}")
        print(f"{'='*60}")
        
        return successful, failed
    
    # ==================== STEP 4: CREATE DICTIONARY ====================
    
    def create_dictionary(self, parsed_words):
        """Create dictionary.txt mapping"""
        print("\n" + "="*60)
        print("STEP 4: CREATING DICTIONARY")
        print("="*60)
        
        dictionary_entries = []
        seen_words = {}
        
        for idx, word_info in enumerate(parsed_words):
            class_id = f"{idx:06d}"
            word = word_info['word']
            dictionary_entries.append(f"{class_id}\t{word}")
            
            if word not in seen_words:
                seen_words[word] = []
            seen_words[word].append(class_id)
        
        # Write dictionary
        output_file = os.path.join(self.dataset_dir, 'dictionary.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dictionary_entries))
        
        print(f"\n[OK] Dictionary created: {output_file}")
        print(f"  Total entries: {len(dictionary_entries)}")
        print(f"  Unique words: {len(seen_words)}")
        
        # Show words with multiple videos
        multi_video = {w: ids for w, ids in seen_words.items() if len(ids) > 1}
        if multi_video:
            print(f"\n  Words with multiple videos: {len(multi_video)}")
            for word, ids in list(multi_video.items())[:3]:
                print(f"    - {word}: {len(ids)} videos")
        
        return output_file
    
    # ==================== MAIN PIPELINE ====================
    
    def prepare_dataset(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("QIPEDC VIETNAMESE SIGN LANGUAGE DATASET PREPARATION".center(70))
        print("="*70)
        
        try:
            # Step 1: Crawl
            words_data = self.crawl_qipedc()
            if not words_data:
                print("\n[ERROR] Failed to crawl data")
                return False
            
            # Step 2: Parse
            parsed_words = self.parse_data(words_data)
            if not parsed_words:
                print("\n[ERROR] Failed to parse data")
                return False
            
            # Step 3: Download & Extract
            successful, failed = self.download_and_process(parsed_words)
            
            # Step 4: Create Dictionary
            dict_file = self.create_dictionary(parsed_words)
            
            # Final Summary
            print("\n" + "="*70)
            print("DATASET PREPARATION COMPLETE!".center(70))
            print("="*70)
            print(f"\nDataset Statistics:")
            print(f"   - Total words: {len(parsed_words)}")
            print(f"   - Unique words: {len(set(w['word'] for w in parsed_words))}")
            print(f"   - Videos downloaded: {successful}")
            print(f"   - Dictionary: {dict_file}")
            print(f"\nDataset Location:")
            print(f"   - Videos: {self.videos_dir}")
            print(f"   - Frames: {self.frames_dir}")
            print(f"\n[OK] Ready for training!")
            print(f"   Run: python VSL_Isolated_Conv3D.py")
            print("="*70)
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n[WARNING] Process interrupted by user")
            return False
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    preparer = QIPEDCDatasetPreparer()
    preparer.prepare_dataset()
