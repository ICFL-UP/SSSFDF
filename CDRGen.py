# ==============================================================================
# Developed by Vusi Mohlatswa
# ENHANCED CDR GENERATOR - WITH CUSTOMER INFORMATION
# Version 3.1 - Adds customer name and address that matches the tower location
# ==============================================================================

import csv
import random
import datetime
import numpy as np
from collections import defaultdict, deque
from enum import Enum
import math
import hashlib
import json
import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import sys

# ===================================================================
# CONFIGURATION OF 
# ===================================================================

class Config:
    """Central configuration with sensible defaults and environment overrides"""
    
    # Core Settings
    NUM_RECORDS = 1048575  # Reduced for testing, change to 1000000 for full run
    OUTPUT_FILE = "lower_cdr_records.csv"
    REAL_WORLD_FRAUD_RATIO = 0.10
    TRAINING_FRAUD_RATIO = 0.10
    FRAUD_RATIO = TRAINING_FRAUD_RATIO
    RANDOM_SEED = 42
    
    # Fraud Distribution
    SIM_SWAP_PROBABILITY = 0.25
    SUBSCRIPTION_FRAUD_PROBABILITY = 0.25
    CALL_FORWARDING_FRAUD_PROBABILITY = 0.20
    PREMIUM_SMS_FRAUD_PROBABILITY = 0.15
    DATA_ROAMING_FRAUD_PROBABILITY = 0.15
    
    # Temporal Patterns
    GENERATION_DAYS_BACK = 90
    PEAK_HOURS_MORNING = (8, 10)
    PEAK_HOURS_AFTERNOON = (17, 20)
    LATE_NIGHT_HOURS = (23, 5)
    
    # Device Settings
    LEGITIMATE_DEVICE_CHANGE_RATE = 0.02
    UPGRADE_CYCLE_FLAGSHIP = 365
    UPGRADE_CYCLE_MIDRANGE = 730
    UPGRADE_CYCLE_BUDGET = 1095
    
    # Performance Settings
    BATCH_SIZE = 1000
    NUM_THREADS = 4
    ENABLE_PARALLEL = False
    ENABLE_STREAMING = False
    PRESERVE_CHRONOLOGY = True
    
    # Privacy Settings
    ENABLE_DIFFERENTIAL_PRIVACY = False
    PRIVACY_EPSILON = 1.0
    ANONYMIZE_SENSITIVE = True
    
    # Quality Settings
    ENABLE_VALIDATION = True
    VALIDATION_SPLIT = 0.05

# Initialize logging - use UTF-8 encoding for console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===================================================================
# ENUMS AND ENHANCED DATA CLASSES
# ===================================================================

class FraudType(Enum):
    LEGITIMATE = "LEGITIMATE"
    SIM_SWAP = "SIM_SWAP"
    SUBSCRIPTION_FRAUD = "SUBSCRIPTION_FRAUD"
    CALL_FORWARDING = "CALL_FORWARDING_FRAUD"
    PREMIUM_SMS = "PREMIUM_SMS_FRAUD"
    DATA_ROAMING = "DATA_ROAMING_FRAUD"
    CLONING = "CLONING_FRAUD"
    MONEY_LAUNDERING = "MONEY_LAUNDERING_FRAUD"
    IOT_BOTNET = "IOT_BOTNET_FRAUD"
    DEEPFAKE_AUDIO = "DEEPFAKE_AUDIO_FRAUD"

class FraudSophistication(Enum):
    NAIVE = "NAIVE"
    MODERATE = "MODERATE"
    ADVANCED = "ADVANCED"
    EVOLVING = "EVOLVING"

class EventType(Enum):
    VOICE_CALL = "VOICE_CALL"
    SMS = "SMS"
    VIDEO_CALL = "VIDEO_CALL"
    GPRS = "GPRS"
    MMS = "MMS"
    DATA_SESSION = "DATA_SESSION"
    MONEY_TRANSFER = "MONEY_TRANSFER"
    AIRTIME_PURCHASE = "AIRTIME_PURCHASE"
    ROAMING_CALL = "ROAMING_CALL"
    FORWARDED_CALL = "FORWARDED_CALL"

class TowerEnvironment(Enum):
    URBAN = "URBAN"
    SUBURBAN = "SUBURBAN"
    RURAL = "RURAL"
    INDUSTRIAL = "INDUSTRIAL"
    INDOOR = "INDOOR"
    ROAMING = "ROAMING"

class SubscriberSegment(Enum):
    HIGH_VALUE = "HIGH_VALUE"
    BUSINESS = "BUSINESS"
    AVERAGE = "AVERAGE"
    BUDGET = "BUDGET"
    NEW = "NEW"
    ROAMING = "ROAMING"
    IOT = "IOT"

@dataclass
class DeviceInfo:
    """Enhanced device information with upgrade tracking"""
    imei: str
    imsi: str
    device_name: str
    device_brand: str
    tac: str
    assigned_at: datetime.datetime
    device_type: str
    price_range: str
    upgrade_cycle_days: int
    subscription_date: datetime.datetime
    sim_change_date: Optional[datetime.datetime] = None
    is_sim_swapped: bool = False
    
    def is_due_for_upgrade(self, current_time: datetime.datetime) -> bool:
        elapsed_days = (current_time - self.assigned_at).days
        return elapsed_days > self.upgrade_cycle_days

@dataclass
class SocialContact:
    """Social graph contact with interaction strength"""
    msisdn: str
    frequency: float
    last_interaction: Optional[datetime.datetime] = None
    interaction_type: str = "REGULAR"
    strength: float = 0.5

@dataclass
class SubscriberProfile:
    """Enhanced subscriber profile with behavioral patterns"""
    msisdn: str
    segment: SubscriberSegment
    current_device: Optional[DeviceInfo] = None
    device_history: List[DeviceInfo] = field(default_factory=list)
    social_contacts: List[SocialContact] = field(default_factory=list)
    call_behavior: Dict = field(default_factory=dict)
    temporal_patterns: Dict = field(default_factory=dict)
    risk_flags: List[Dict] = field(default_factory=list)
    creation_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_activity: Optional[datetime.datetime] = None
    fraud_history: List[Dict] = field(default_factory=list)
    customer_segment: str = "AVERAGE"
    monthly_spend: float = 0.0
    loyalty_score: float = 0.5
    full_name: str = ""
    address: str = ""

# ===================================================================
# TOWER DATABASE
# ===================================================================

class EnhancedTowerDatabase:
    """Enhanced tower database with roaming support"""
    
    def __init__(self):
        self.towers = self._init_towers()
        self.roaming_towers = self._init_roaming_towers()
        self.tower_adjacency = self._build_adjacency()
        
    def _init_towers(self):
        towers = [
            {"tower_id": "TWR1001", "name": "JHB-CBD-01", "lac": "1001", "cell_id": "CELL001", 
             "latitude": -26.2041, "longitude": 28.0473, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR1002", "name": "JHB-Sandton-01", "lac": "1002", "cell_id": "CELL002", 
             "latitude": -26.1076, "longitude": 28.0567, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR1003", "name": "JHB-Rosebank-01", "lac": "1003", "cell_id": "CELL003", 
             "latitude": -26.1470, "longitude": 28.0345, "tower_type": "MICRO", "environment": "URBAN"},
            {"tower_id": "TWR1004", "name": "JHB-Midrand-01", "lac": "1004", "cell_id": "CELL004", 
             "latitude": -25.9985, "longitude": 28.1273, "tower_type": "MACRO", "environment": "SUBURBAN"},
            {"tower_id": "TWR1005", "name": "JHB-Soweto-01", "lac": "1005", "cell_id": "CELL005", 
             "latitude": -26.2485, "longitude": 27.8580, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR1006", "name": "JHB-Randburg-01", "lac": "1006", "cell_id": "CELL006", 
             "latitude": -26.1009, "longitude": 28.0017, "tower_type": "MICRO", "environment": "URBAN"},
            {"tower_id": "TWR1007", "name": "JHB-Bedfordview-01", "lac": "1007", "cell_id": "CELL007", 
             "latitude": -26.1782, "longitude": 28.1351, "tower_type": "MICRO", "environment": "SUBURBAN"},
            {"tower_id": "TWR2001", "name": "CPT-CBD-01", "lac": "2001", "cell_id": "CELL008", 
             "latitude": -33.9249, "longitude": 18.4241, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR2002", "name": "CPT-SeaPoint-01", "lac": "2002", "cell_id": "CELL009", 
             "latitude": -33.9153, "longitude": 18.3855, "tower_type": "MICRO", "environment": "URBAN"},
            {"tower_id": "TWR2003", "name": "CPT-Claremont-01", "lac": "2003", "cell_id": "CELL010", 
             "latitude": -33.9808, "longitude": 18.4660, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR3001", "name": "DBN-CBD-01", "lac": "3001", "cell_id": "CELL011", 
             "latitude": -29.8587, "longitude": 31.0218, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR4001", "name": "PTA-CBD-01", "lac": "4001", "cell_id": "CELL012", 
             "latitude": -25.7479, "longitude": 28.2293, "tower_type": "MACRO", "environment": "URBAN"},
            {"tower_id": "TWR5001", "name": "RURAL-Mpumalanga-01", "lac": "5001", "cell_id": "CELL013", 
             "latitude": -25.4658, "longitude": 30.9854, "tower_type": "MACRO", "environment": "RURAL"},
        ]
        return towers
    
    def _init_roaming_towers(self):
        return [
            {"tower_id": "ROAM001", "name": "UK-London-01", "lac": "ROAM001", "cell_id": "RCELL001",
             "latitude": 51.5074, "longitude": -0.1278, "tower_type": "ROAMING", "environment": "ROAMING"},
            {"tower_id": "ROAM002", "name": "USA-NY-01", "lac": "ROAM002", "cell_id": "RCELL002",
             "latitude": 40.7128, "longitude": -74.0060, "tower_type": "ROAMING", "environment": "ROAMING"},
        ]
    
    def _build_adjacency(self):
        adjacency = {}
        all_towers = self.towers + self.roaming_towers
        for tower in all_towers:
            tower_id = tower["tower_id"]
            adjacency[tower_id] = []
            for other in all_towers:
                if other["tower_id"] != tower_id:
                    dist = self.calculate_distance(
                        tower["latitude"], tower["longitude"],
                        other["latitude"], other["longitude"]
                    )
                    if dist < 50:
                        adjacency[tower_id].append({
                            "tower_id": other["tower_id"],
                            "distance": dist
                        })
        return adjacency
    
    def get_random_tower(self):
        return random.choice(self.towers)
    
    def get_roaming_tower(self):
        return random.choice(self.roaming_towers)
    
    def get_towers_by_environment(self, environment):
        return [t for t in self.towers if t["environment"] == environment]
    
    def get_adjacent_towers(self, tower_id, max_distance=50):
        if tower_id in self.tower_adjacency:
            return [t for t in self.tower_adjacency[tower_id] 
                   if t["distance"] <= max_distance]
        return []
    
    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

TOWER_DB = EnhancedTowerDatabase()

# ===================================================================
# CUSTOMER NAME AND ADDRESS GENERATION
# ===================================================================

FIRST_NAMES = [
    "Thabo", "Lerato", "Naledi", "Sipho", "Zanele", "Kagiso", "Tshepo", "Amanda",
    "Michael", "David", "Mary", "Peter", "Sarah", "John", "Elizabeth", "William",
    "Andre", "Marlize", "Hendrik", "Annelie", "Pieter", "Helena", "Jacobus", "Magdalena",
    "Ayanda", "Buhle", "Lindiwe", "Sibusiso", "Thandi", "Nomsa", "Phindile", "Themba"
]

LAST_NAMES = [
    "Mkhize", "Nkosi", "Smith", "Botha", "Van der Merwe", "Naidoo", "Govender",
    "Coetzee", "De Beer", "Kruger", "Mokwena", "Mthembu", "Ndlovu", "Zulu",
    "Pretorius", "Swanepoel", "Venter", "Fourie", "Du Plessis", "Van Wyk",
    "Pillay", "Reddy", "Singh", "Moodley", "Naicker", "Bhana", "Maharaj"
]

STREET_NAMES = [
    "Main Street", "Church Street", "Queen Street", "King Street", "Station Road",
    "Beach Road", "Sunset Avenue", "Oak Avenue", "Pine Road", "Maple Drive",
    "High Street", "Park Lane", "Garden Street", "Riverside Drive", "Hilltop Road",
    "Central Avenue", "West Street", "East Street", "South Road", "North Street"
]

CITY_MAP = {
    "JHB": {"city": "Johannesburg", "province": "Gauteng", "suburbs": ["Sandton", "Rosebank", "Randburg", "Midrand", "Soweto", "Bedfordview", "Bryanston"]},
    "CPT": {"city": "Cape Town", "province": "Western Cape", "suburbs": ["Sea Point", "Claremont", "Cape Town CBD", "Constantia", "Rondebosch", "Newlands"]},
    "DBN": {"city": "Durban", "province": "KwaZulu-Natal", "suburbs": ["Durban CBD", "Umhlanga", "Ballito", "Westville", "Amanzimtoti"]},
    "PTA": {"city": "Pretoria", "province": "Gauteng", "suburbs": ["Pretoria CBD", "Hatfield", "Brooklyn", "Menlo Park", "Sunnyside"]},
    "RURAL": {"city": "Nelspruit", "province": "Mpumalanga", "suburbs": ["Nelspruit CBD", "Riverside", "Mataffin"]},
}

def generate_customer_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def generate_customer_address(tower_name: str) -> str:
    """
    Generate a plausible physical address based on the tower's city.
    tower_name format: e.g., 'JHB-CBD-01' -> prefix 'JHB'
    """
    if not tower_name:
        city_info = CITY_MAP["JHB"]  # fallback
    else:
        prefix = tower_name.split('-')[0]
        city_info = CITY_MAP.get(prefix, CITY_MAP["JHB"])
    street_num = random.randint(1, 999)
    street = random.choice(STREET_NAMES)
    suburb = random.choice(city_info["suburbs"])
    city = city_info["city"]
    province = city_info["province"]
    return f"{street_num} {street}, {suburb}, {city}, {province}"

# ===================================================================
# DEVICE DATABASE
# ===================================================================

class EnhancedDeviceDatabase:
    """Enhanced device database with upgrade paths"""
    
    def __init__(self):
        self.devices = self._init_devices()
        self.brand_upgrade_paths = self._build_upgrade_paths()
        self.price_tiers = {
            "FLAGSHIP": ["Samsung Galaxy S24 Ultra", "iPhone 15 Pro Max"],
            "PREMIUM": ["Samsung Galaxy S24 Plus", "iPhone 15 Pro"],
            "MID_RANGE": ["Samsung Galaxy A55", "iPhone 15"],
            "BUDGET": ["Samsung Galaxy A15", "Nokia G60"]
        }
    
    def _init_devices(self):
        return [
            {"name": "Samsung Galaxy S24 Ultra", "tac": "35265411", "brand": "Samsung", "tier": "FLAGSHIP"},
            {"name": "Samsung Galaxy S24 Plus", "tac": "35265410", "brand": "Samsung", "tier": "PREMIUM"},
            {"name": "Samsung Galaxy S24", "tac": "35265409", "brand": "Samsung", "tier": "PREMIUM"},
            {"name": "Samsung Galaxy S23 Ultra", "tac": "35481720", "brand": "Samsung", "tier": "FLAGSHIP"},
            {"name": "Samsung Galaxy S23 Plus", "tac": "35481719", "brand": "Samsung", "tier": "PREMIUM"},
            {"name": "Samsung Galaxy A55", "tac": "35482414", "brand": "Samsung", "tier": "MID_RANGE"},
            {"name": "Samsung Galaxy A35", "tac": "35482415", "brand": "Samsung", "tier": "MID_RANGE"},
            {"name": "Samsung Galaxy A15", "tac": "35482416", "brand": "Samsung", "tier": "BUDGET"},
            {"name": "iPhone 15 Pro Max", "tac": "35266385", "brand": "Apple", "tier": "FLAGSHIP"},
            {"name": "iPhone 15 Pro", "tac": "35266384", "brand": "Apple", "tier": "PREMIUM"},
            {"name": "iPhone 15", "tac": "35266382", "brand": "Apple", "tier": "PREMIUM"},
            {"name": "iPhone 14 Pro Max", "tac": "35265422", "brand": "Apple", "tier": "FLAGSHIP"},
            {"name": "iPhone 14", "tac": "35265420", "brand": "Apple", "tier": "MID_RANGE"},
            {"name": "Google Pixel 8 Pro", "tac": "35266391", "brand": "Google", "tier": "FLAGSHIP"},
            {"name": "Google Pixel 8", "tac": "35266390", "brand": "Google", "tier": "PREMIUM"},
            {"name": "Xiaomi 14 Pro", "tac": "35270312", "brand": "Xiaomi", "tier": "FLAGSHIP"},
            {"name": "Xiaomi 14", "tac": "35270311", "brand": "Xiaomi", "tier": "PREMIUM"},
            {"name": "Xiaomi Redmi Note 13", "tac": "35270310", "brand": "Xiaomi", "tier": "MID_RANGE"},
            {"name": "OnePlus 12", "tac": "35265426", "brand": "OnePlus", "tier": "FLAGSHIP"},
            {"name": "Tecno Camon 20 Pro", "tac": "35270324", "brand": "Tecno", "tier": "MID_RANGE"},
            {"name": "Tecno Spark 10", "tac": "35270325", "brand": "Tecno", "tier": "BUDGET"},
            {"name": "Infinix Zero 30", "tac": "35270327", "brand": "Infinix", "tier": "MID_RANGE"},
            {"name": "Infinix Note 30", "tac": "35270328", "brand": "Infinix", "tier": "BUDGET"},
        ]
    
    def _build_upgrade_paths(self):
        return {
            "Samsung": [
                ("Samsung Galaxy A15", "Samsung Galaxy A35", 365),
                ("Samsung Galaxy A35", "Samsung Galaxy A55", 365),
                ("Samsung Galaxy S23", "Samsung Galaxy S24", 365),
                ("Samsung Galaxy S23 Ultra", "Samsung Galaxy S24 Ultra", 365),
            ],
            "Apple": [
                ("iPhone SE", "iPhone 14", 365),
                ("iPhone 14", "iPhone 15", 365),
                ("iPhone 14 Pro", "iPhone 15 Pro", 365),
                ("iPhone 14 Pro Max", "iPhone 15 Pro Max", 365),
            ]
        }
    
    def get_random_device(self):
        return random.choice(self.devices)
    
    def get_different_device(self, current_device: Dict) -> Dict:
        """Get a device different from the current one"""
        devices = [d for d in self.devices if d["name"] != current_device["name"]]
        if devices:
            return random.choice(devices)
        return self.get_random_device()
    
    def get_next_in_upgrade_path(self, current_device, time_elapsed_days):
        brand = current_device["brand"]
        current_name = current_device["name"]
        
        if brand not in self.brand_upgrade_paths:
            return None
        
        paths = self.brand_upgrade_paths.get(brand, [])
        upgrade_threshold = 365 if current_device.get("tier") in ["FLAGSHIP", "PREMIUM"] else 730
        
        if time_elapsed_days > upgrade_threshold:
            for path in paths:
                if path[0] == current_name and time_elapsed_days > path[2]:
                    new_device = next((d for d in self.devices if d["name"] == path[1]), None)
                    if new_device:
                        return new_device
        return None
    
    def get_devices_by_tier(self, tier):
        return [d for d in self.devices if d.get("tier") == tier]

DEVICE_DB = EnhancedDeviceDatabase()

# ===================================================================
# SOCIAL GRAPH MANAGER
# ===================================================================

class SocialGraphManager:
    """Manages subscriber social networks"""
    
    def __init__(self):
        self.graph = defaultdict(set)
        self.contact_frequencies = defaultdict(lambda: defaultdict(float))
        self.communities = defaultdict(set)
    
    def build_social_network(self, subscribers: List[str], num_contacts_range=(5, 20)):
        if not subscribers:
            return
        
        num_communities = max(1, len(subscribers) // 50)
        
        for i, subscriber in enumerate(subscribers):
            community_id = i % num_communities
            self.communities[community_id].add(subscriber)
        
        for community_id, members in self.communities.items():
            members_list = list(members)
            for member in members_list:
                num_contacts = random.randint(3, min(5, len(members_list) - 1))
                if len(members_list) > 1:
                    contacts = random.sample(
                        [m for m in members_list if m != member], 
                        min(num_contacts, len(members_list) - 1)
                    )
                    for contact in contacts:
                        self.graph[member].add(contact)
                        self.contact_frequencies[member][contact] = random.uniform(0.3, 0.6)
        
        # Add weak ties
        for subscriber in subscribers:
            if len(self.graph.get(subscriber, set())) < 5:
                all_others = [s for s in subscribers if s != subscriber]
                if all_others:
                    contacts = random.sample(all_others, min(2, len(all_others)))
                    for contact in contacts:
                        self.graph[subscriber].add(contact)
                        self.contact_frequencies[subscriber][contact] = random.uniform(0.01, 0.1)
        
        return self
    
    def get_contacts(self, subscriber: str) -> List[str]:
        return list(self.graph.get(subscriber, set()))
    
    def get_weighted_contact(self, subscriber: str) -> Optional[str]:
        contacts = self.get_contacts(subscriber)
        if not contacts:
            return None
        
        weights = [self.contact_frequencies[subscriber].get(c, 0.1) for c in contacts]
        total = sum(weights)
        if total == 0:
            return random.choice(contacts)
        
        normalized_weights = [w / total for w in weights]
        return np.random.choice(contacts, p=normalized_weights)

# ===================================================================
# ENHANCED DEVICE ASSIGNMENT TRACKER WITH SIM SWAP LOGIC
# ===================================================================

class EnhancedDeviceAssignmentTracker:
    """Enhanced device tracking with proper SIM swap logic showing two distinct devices"""
    
    def __init__(self):
        self.subscriber_devices: Dict[str, DeviceInfo] = {}
        self.device_history: Dict[str, List[DeviceInfo]] = defaultdict(list)
        self.sim_swap_count: Dict[str, int] = defaultdict(int)
        self.subscriber_profiles: Dict[str, SubscriberProfile] = {}
        self.upgrade_events: List[Dict] = []
        self.sim_swap_events: List[Dict] = []
        self.sim_swap_device_pairs: Dict[str, List[Tuple[DeviceInfo, DeviceInfo]]] = defaultdict(list)
        
    def get_or_assign_device(self, msisdn: str, allow_swap: bool = False, 
                            swap_time: Optional[datetime.datetime] = None,
                            segment: SubscriberSegment = SubscriberSegment.AVERAGE,
                            subscription_date: Optional[datetime.datetime] = None) -> DeviceInfo:
        """
        Get or assign device with proper SIM swap tracking.
        When allow_swap=True, creates a new device with different IMEI and IMSI.
        The swap time is recorded as the LAST_SIM_CHANGE_DATE.
        """
        
        if msisdn not in self.subscriber_devices:
            # Initial subscription
            if subscription_date is None:
                subscription_date = random_subscription_datetime()
            
            device = self._generate_device(segment)
            device_info = self._create_device_info(
                device, 
                datetime.datetime.now(), 
                segment,
                subscription_date=subscription_date,
                sim_change_date=subscription_date,
                is_sim_swapped=False
            )
            self.subscriber_devices[msisdn] = device_info
            self.device_history[msisdn].append(device_info)
            self._initialize_profile(msisdn, segment)
            return device_info
        
        current = self.subscriber_devices[msisdn]
        
        # Check for device upgrade (not SIM swap)
        if self._should_upgrade(current, datetime.datetime.now()):
            upgrade_device = self._get_upgrade_device(current)
            if upgrade_device:
                upgrade_info = self._create_device_info(
                    upgrade_device, 
                    datetime.datetime.now(), 
                    self.subscriber_profiles[msisdn].segment,
                    subscription_date=current.subscription_date,
                    sim_change_date=current.sim_change_date,
                    is_sim_swapped=False
                )
                self.subscriber_devices[msisdn] = upgrade_info
                self.device_history[msisdn].append(upgrade_info)
                self.upgrade_events.append({
                    "msisdn": msisdn,
                    "old_device": current.device_name,
                    "old_imei": current.imei,
                    "old_imsi": current.imsi,
                    "new_device": upgrade_info.device_name,
                    "new_imei": upgrade_info.imei,
                    "new_imsi": upgrade_info.imsi,
                    "timestamp": datetime.datetime.now(),
                    "type": "DEVICE_UPGRADE"
                })
                return upgrade_info
        
        if allow_swap:
            # SIM SWAP - Different IMEI and IMSI, new device
            # Store the old device info before swap
            old_device_info = DeviceInfo(
                imei=current.imei,
                imsi=current.imsi,
                device_name=current.device_name,
                device_brand=current.device_brand,
                tac=current.tac,
                assigned_at=current.assigned_at,
                device_type=current.device_type,
                price_range=current.price_range,
                upgrade_cycle_days=current.upgrade_cycle_days,
                subscription_date=current.subscription_date,
                sim_change_date=current.sim_change_date,
                is_sim_swapped=current.is_sim_swapped
            )
            
            # Generate new device - must be different from current
            new_device = self._generate_device(self.subscriber_profiles[msisdn].segment)
            new_imei = self._generate_imei(new_device["tac"])
            
            # Ensure different IMEI from current one
            attempts = 0
            while new_imei == current.imei and attempts < 10:
                new_device = DEVICE_DB.get_different_device(new_device)
                new_imei = self._generate_imei(new_device["tac"])
                attempts += 1
            
            # Ensure different IMSI
            new_imsi = self._generate_imsi()
            while new_imsi == current.imsi:
                new_imsi = self._generate_imsi()
            
            swap_time = swap_time or datetime.datetime.now()
            
            # Create new device info with SIM swap info
            swap_info = self._create_device_info_from_device(
                new_device, 
                new_imei, 
                swap_time, 
                self.subscriber_profiles[msisdn].segment,
                subscription_date=current.subscription_date,
                sim_change_date=swap_time,
                is_sim_swapped=True
            )
            
            # Override the IMSI to ensure it's different
            swap_info.imsi = new_imsi
            
            # Store the device pair (before and after swap)
            self.sim_swap_device_pairs[msisdn].append((old_device_info, swap_info))
            
            # Update current device
            self.subscriber_devices[msisdn] = swap_info
            self.device_history[msisdn].append(swap_info)
            self.sim_swap_count[msisdn] += 1
            
            # Track detailed SIM swap event
            swap_event = {
                "msisdn": msisdn,
                "swap_number": self.sim_swap_count[msisdn],
                "swap_time": swap_time.strftime("%Y-%m-%d %H:%M:%S"),
                "subscription_date": current.subscription_date.strftime("%Y-%m-%d %H:%M:%S"),
                "old_device_name": old_device_info.device_name,
                "old_device_brand": old_device_info.device_brand,
                "old_imei": old_device_info.imei,
                "old_imsi": old_device_info.imsi,
                "new_device_name": swap_info.device_name,
                "new_device_brand": swap_info.device_brand,
                "new_imei": swap_info.imei,
                "new_imsi": swap_info.imsi,
                "device_changed": old_device_info.device_name != swap_info.device_name,
                "imei_changed": old_device_info.imei != swap_info.imei,
                "imsi_changed": old_device_info.imsi != swap_info.imsi
            }
            self.sim_swap_events.append(swap_event)
            
            # Add risk flag to profile
            self.subscriber_profiles[msisdn].risk_flags.append({
                "type": "SIM_SWAP",
                "timestamp": swap_time,
                "swap_number": self.sim_swap_count[msisdn],
                "old_imei": old_device_info.imei,
                "new_imei": swap_info.imei,
                "old_imsi": old_device_info.imsi,
                "new_imsi": swap_info.imsi,
                "old_device": old_device_info.device_name,
                "new_device": swap_info.device_name,
                "subscription_date": current.subscription_date
            })
            
            return swap_info
        
        return current
    
    def _create_device_info(self, device: Dict, assigned_at: datetime.datetime, 
                          segment: SubscriberSegment,
                          subscription_date: Optional[datetime.datetime] = None,
                          sim_change_date: Optional[datetime.datetime] = None,
                          is_sim_swapped: bool = False) -> DeviceInfo:
        imei = self._generate_imei(device["tac"])
        imsi = self._generate_imsi()
        return self._create_device_info_from_device(
            device, imei, assigned_at, segment, 
            subscription_date=subscription_date,
            sim_change_date=sim_change_date,
            is_sim_swapped=is_sim_swapped
        )
    
    def _create_device_info_from_device(self, device: Dict, imei: str, assigned_at: datetime.datetime,
                                      segment: SubscriberSegment,
                                      subscription_date: Optional[datetime.datetime] = None,
                                      sim_change_date: Optional[datetime.datetime] = None,
                                      is_sim_swapped: bool = False) -> DeviceInfo:
        imsi = self._generate_imsi()
        tier = device.get("tier", "MID_RANGE")
        
        cycle_map = {
            "FLAGSHIP": Config.UPGRADE_CYCLE_FLAGSHIP,
            "PREMIUM": Config.UPGRADE_CYCLE_FLAGSHIP,
            "MID_RANGE": Config.UPGRADE_CYCLE_MIDRANGE,
            "BUDGET": Config.UPGRADE_CYCLE_BUDGET
        }
        upgrade_cycle = cycle_map.get(tier, Config.UPGRADE_CYCLE_MIDRANGE)
        
        if segment == SubscriberSegment.HIGH_VALUE:
            upgrade_cycle = int(upgrade_cycle * 0.7)
        elif segment == SubscriberSegment.BUDGET:
            upgrade_cycle = int(upgrade_cycle * 1.3)
        
        if subscription_date is None:
            subscription_date = random_subscription_datetime()
        
        if sim_change_date is None:
            sim_change_date = subscription_date
        
        return DeviceInfo(
            imei=imei,
            imsi=imsi,
            device_name=device["name"],
            device_brand=device["brand"],
            tac=device["tac"],
            assigned_at=assigned_at,
            device_type=tier,
            price_range=tier,
            upgrade_cycle_days=upgrade_cycle,
            subscription_date=subscription_date,
            sim_change_date=sim_change_date,
            is_sim_swapped=is_sim_swapped
        )
    
    def _should_upgrade(self, device_info: DeviceInfo, current_time: datetime.datetime) -> bool:
        elapsed_days = (current_time - device_info.assigned_at).days
        return elapsed_days > device_info.upgrade_cycle_days
    
    def _get_upgrade_device(self, current: DeviceInfo) -> Optional[Dict]:
        current_device = next(
            (d for d in DEVICE_DB.devices if d["name"] == current.device_name), 
            None
        )
        if not current_device:
            return None
        
        next_device = DEVICE_DB.get_next_in_upgrade_path(
            current_device, 
            (datetime.datetime.now() - current.assigned_at).days
        )
        
        if next_device:
            return next_device
        
        same_brand = [d for d in DEVICE_DB.devices if d["brand"] == current.device_brand]
        if same_brand and len(same_brand) > 1:
            return random.choice([d for d in same_brand if d["name"] != current.device_name])
        
        return DEVICE_DB.get_random_device()
    
    def _generate_device(self, segment: SubscriberSegment) -> Dict:
        if segment == SubscriberSegment.HIGH_VALUE:
            tier = random.choice(["FLAGSHIP", "PREMIUM"])
        elif segment == SubscriberSegment.BUSINESS:
            tier = random.choice(["PREMIUM", "MID_RANGE"])
        elif segment == SubscriberSegment.NEW:
            tier = random.choice(["MID_RANGE", "BUDGET"])
        else:
            tier = random.choice(["MID_RANGE", "BUDGET", "PREMIUM"])
        
        devices = DEVICE_DB.get_devices_by_tier(tier)
        if devices:
            return random.choice(devices)
        return DEVICE_DB.get_random_device()
    
    def _generate_imei(self, tac: str) -> str:
        snr = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        imei_without_check = f"{tac}{snr}"
        check_digit = self._calculate_luhn_check_digit(imei_without_check)
        return f"{tac}{snr}{check_digit}"
    
    def _calculate_luhn_check_digit(self, number: str) -> int:
        digits = [int(d) for d in number]
        for i in range(len(digits) - 1, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] = digits[i] - 9
        total = sum(digits)
        check_digit = (10 - (total % 10)) % 10
        return check_digit
    
    def _generate_imsi(self) -> str:
        mcc = random.choice(["655", "310", "311", "312", "313", "620", "618", "639"])
        if mcc == "655":
            mnc = random.choice(["01", "02", "03", "04", "05", "06"])
        else:
            mnc = random.choice(["010", "020", "030", "040", "050", "060"])
        msn_length = 15 - len(mcc) - len(mnc)
        msn = ''.join([str(random.randint(0, 9)) for _ in range(msn_length)])
        return f"{mcc}{mnc}{msn}"
    
    def _initialize_profile(self, msisdn: str, segment: SubscriberSegment):
        self.subscriber_profiles[msisdn] = SubscriberProfile(
            msisdn=msisdn,
            segment=segment,
            creation_date=datetime.datetime.now(),
            monthly_spend=random.uniform(50, 500),
            loyalty_score=random.uniform(0.3, 0.9)
        )
    
    def get_subscriber_subscription_date(self, msisdn: str) -> Optional[datetime.datetime]:
        if msisdn in self.subscriber_devices:
            return self.subscriber_devices[msisdn].subscription_date
        return None
    
    def get_subscriber_sim_change_date(self, msisdn: str) -> Optional[datetime.datetime]:
        if msisdn in self.subscriber_devices:
            return self.subscriber_devices[msisdn].sim_change_date
        return None
    
    def get_subscriber_device_history(self, msisdn: str) -> List[DeviceInfo]:
        return self.device_history.get(msisdn, [])
    
    def get_sim_swap_pairs(self, msisdn: str) -> List[Tuple[DeviceInfo, DeviceInfo]]:
        return self.sim_swap_device_pairs.get(msisdn, [])

# ===================================================================
# TEMPORAL PATTERN GENERATOR
# ===================================================================

class TemporalPatternGenerator:
    """Generates realistic temporal patterns for subscriber behavior"""
    
    def __init__(self):
        self.weekday_patterns = {
            "BUSINESS": {
                "weekday": [0.3] * 7 + [1.5] * 6 + [2.0] * 4 + [1.5] * 6 + [0.3],
                "weekend": [0.1] * 10 + [1.5] * 6 + [2.0] * 4 + [1.5] * 4
            },
            "AVERAGE": {
                "weekday": [0.5] * 6 + [1.5] * 4 + [1.0] * 4 + [1.5] * 4 + [1.0] * 6,
                "weekend": [0.3] * 8 + [1.5] * 4 + [2.0] * 4 + [1.5] * 4 + [0.5]
            }
        }
        self.seasonal_factors = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 0.9,
                                 7: 0.9, 8: 1.0, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.3}
    
    def get_temporal_multiplier(self, dt: datetime.datetime, segment: str = "AVERAGE") -> float:
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        month = dt.month
        
        pattern_type = "weekend" if is_weekend else "weekday"
        segment_pattern = self.weekday_patterns.get(segment, self.weekday_patterns["AVERAGE"])
        if hour < len(segment_pattern[pattern_type]):
            base_multiplier = segment_pattern[pattern_type][hour]
        else:
            base_multiplier = 0.5
        
        seasonal_factor = self.seasonal_factors.get(month, 1.0)
        random_variation = 0.8 + 0.4 * random.random()
        
        return base_multiplier * seasonal_factor * random_variation

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def random_msisdn() -> str:
    return f"+277{random.randint(10000000, 99999999)}"

def random_datetime(days_back: int = 30) -> datetime.datetime:
    now = datetime.datetime.now()
    days_ago = random.randint(0, days_back)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)
    return now - datetime.timedelta(days=days_ago, hours=hours, minutes=minutes, seconds=seconds)

def random_temporal_datetime(days_back: int, segment: SubscriberSegment,
                             temporal_patterns: Optional[TemporalPatternGenerator]) -> datetime.datetime:
    """Generate a timestamp whose hour follows the subscriber segment's activity profile."""
    now = datetime.datetime.now()
    days_ago = random.randint(0, max(0, days_back))
    base_date = (now - datetime.timedelta(days=days_ago)).date()

    if temporal_patterns is None:
        hour = random.randint(0, 23)
    else:
        segment_name = "BUSINESS" if segment == SubscriberSegment.BUSINESS else "AVERAGE"
        candidates = [datetime.datetime.combine(base_date, datetime.time(hour=h)) for h in range(24)]
        weights = [max(0.01, temporal_patterns.get_temporal_multiplier(dt, segment_name)) for dt in candidates]
        hour = random.choices(range(24), weights=weights, k=1)[0]

    dt = datetime.datetime.combine(
        base_date,
        datetime.time(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
    )
    return min(dt, now)

def random_subscription_datetime() -> datetime.datetime:
    now = datetime.datetime.now()
    days_ago = random.randint(30, 730)
    return now - datetime.timedelta(days=days_ago, hours=random.randint(0, 23), 
                                   minutes=random.randint(0, 59), seconds=random.randint(0, 59))

def random_sim_change_datetime(subscription_datetime: datetime.datetime) -> datetime.datetime:
    now = datetime.datetime.now()
    lower_bound = subscription_datetime + datetime.timedelta(days=1)
    
    if lower_bound > now:
        return subscription_datetime
    
    if random.random() < 0.70:
        return subscription_datetime
    
    delta_seconds = int((now - lower_bound).total_seconds())
    if delta_seconds < 1:
        return lower_bound
    random_seconds = random.randint(0, delta_seconds)
    return lower_bound + datetime.timedelta(seconds=random_seconds)

def random_data_volume() -> float:
    volumes = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    return round(random.choice(volumes), 2)

def random_charge(duration_sec: int, USAGE_TYPE: str, data_volume: float = 0.0) -> float:
    base_rate = 0.30
    
    if USAGE_TYPE in ["MOC", "MO"]:
        rate = base_rate
    elif USAGE_TYPE in ["MTC", "MT"]:
        rate = base_rate * 0.7
    elif USAGE_TYPE == "SMS":
        rate = 0.20
    elif USAGE_TYPE in ["GPRS", "DATA"]:
        rate = data_volume * 0.05
        return round(max(0.01, rate), 2)
    else:
        rate = base_rate * 0.5
    
    rate *= (0.9 + 0.2 * random.random())
    minutes = duration_sec / 60.0
    charge = round(minutes * rate, 2)
    return max(0.01, charge)

# ===================================================================
# FRAUD REASON GENERATOR - NECESSARY TO ASSIST WITH VALIDATIONS
# ===================================================================

class FraudReasonGenerator:
    """Generates detailed reasons for fraud detection - ASCII only"""
    
    @staticmethod
    def generate_sim_swap_reasons(record: List, device_tracker: EnhancedDeviceAssignmentTracker) -> str:
        """Generate detailed reasons for SIM swap fraud"""
        reasons = []
        msisdn = record[2] if len(record) > 2 else "UNKNOWN"
        
        if msisdn in device_tracker.subscriber_devices:
            device = device_tracker.subscriber_devices[msisdn]
            subscription_date = device.subscription_date
            sim_change_date = device.sim_change_date
            
            now = datetime.datetime.now()
            subscription_age = (now - subscription_date).days if subscription_date else 0
            sim_change_age = (now - sim_change_date).days if sim_change_date else 0
            
            if device.is_sim_swapped:
                reasons.append(f"SIM SWAP DETECTED on {sim_change_date.strftime('%Y-%m-%d')}")
                
                history = device_tracker.get_subscriber_device_history(msisdn)
                if len(history) >= 2:
                    old_device = history[-2]
                    new_device = history[-1]
                    reasons.append(f"DEVICE: {old_device.device_name} -> {new_device.device_name}")
                    reasons.append(f"IMEI: {old_device.imei[:8]}... -> {new_device.imei[:8]}...")
                    reasons.append(f"IMSI: {old_device.imsi[:6]}... -> {new_device.imsi[:6]}...")
                    reasons.append(f"Subscription age: {subscription_age} days")
            
            USAGE_TYPE = record[0] if len(record) > 0 else ""
            duration = int(record[17]) if len(record) > 17 and record[17] else 0
            
            if USAGE_TYPE in ["MOC", "MO"] and duration > 600:
                reasons.append(f"Long call: {duration}s")
            
            CALLED_MSISDN = record[18] if len(record) > 18 else ""
            if CALLED_MSISDN.startswith("+44") or CALLED_MSISDN.startswith("+900"):
                reasons.append(f"Premium number: {CALLED_MSISDN}")
            
            swap_count = device_tracker.sim_swap_count.get(msisdn, 0)
            if swap_count > 1:
                reasons.append(f"Multiple swaps: {swap_count}")
        
        return " | ".join(reasons) if reasons else "SIM swap fraud detected"
    
    @staticmethod
    def generate_subscription_fraud_reasons(record: List, device_tracker: EnhancedDeviceAssignmentTracker) -> str:
        """Generate reasons for subscription fraud"""
        reasons = []
        msisdn = record[2] if len(record) > 2 else "UNKNOWN"
        
        if msisdn in device_tracker.subscriber_devices:
            device = device_tracker.subscriber_devices[msisdn]
            subscription_date = device.subscription_date
            now = datetime.datetime.now()
            subscription_age = (now - subscription_date).days if subscription_date else 0
            
            if subscription_age < 7:
                reasons.append(f"New subscription: {subscription_age} days old")
            elif subscription_age < 30:
                reasons.append(f"Recent subscription: {subscription_age} days old")
        
        USAGE_TYPE = record[0] if len(record) > 0 else ""
        duration = int(record[17]) if len(record) > 17 and record[17] else 0
        
        if USAGE_TYPE in ["MOC", "MO"] and duration > 300:
            reasons.append(f"Long call: {duration}s")
        
        CALLED_MSISDN = record[18] if len(record) > 18 else ""
        if CALLED_MSISDN.startswith("+44") or CALLED_MSISDN.startswith("+900"):
            reasons.append(f"Premium number: {CALLED_MSISDN}")
        
        data_volume = float(record[20]) if len(record) > 20 and record[20] else 0
        if data_volume > 50:
            reasons.append(f"High data: {data_volume}MB")
        
        swap_count = device_tracker.sim_swap_count.get(msisdn, 0)
        if swap_count > 1:
            reasons.append(f"Multiple SIM changes: {swap_count}")
        
        return " | ".join(reasons) if reasons else "Suspicious new account behavior"
    
    @staticmethod
    def generate_legitimate_reason(record: List) -> str:
        return "Normal activity - no fraud detected"
    
    @staticmethod
    def generate_fraud_reason(record: List, fraud_type: str, device_tracker: EnhancedDeviceAssignmentTracker) -> str:
        if fraud_type == "SIM_SWAP" or fraud_type == "SIM_SWAP_EVENT":
            return FraudReasonGenerator.generate_sim_swap_reasons(record, device_tracker)
        elif fraud_type == "SUBSCRIPTION_FRAUD":
            return FraudReasonGenerator.generate_subscription_fraud_reasons(record, device_tracker)
        elif fraud_type == "CALL_FORWARDING_FRAUD":
            return "Call forwarding to premium number detected"
        elif fraud_type == "PREMIUM_SMS_FRAUD":
            return "Premium rate SMS activity detected"
        elif fraud_type == "DATA_ROAMING_FRAUD":
            return "Excessive data usage on roaming network"
        else:
            return "Anomalous behavior pattern detected"

# ===================================================================
# BASE CDR GENERATION
# ===================================================================

def generate_base_cdr(subscriber_msisdn: str = None,
                     device_tracker: EnhancedDeviceAssignmentTracker = None,
                     allow_swap: bool = False,
                     swap_time: datetime.datetime = None,
                     segment: SubscriberSegment = SubscriberSegment.AVERAGE,
                     temporal_patterns: TemporalPatternGenerator = None,
                     social_graph: SocialGraphManager = None,
                     subscription_date: Optional[datetime.datetime] = None) -> List:
    
    if subscriber_msisdn is None:
        subscriber_msisdn = random_msisdn()
    
    record = []
    tower = TOWER_DB.get_random_tower()
    
    is_gprs = random.random() < 0.20
    
    if is_gprs:
        USAGE_TYPEs = ["GPRS", "DATA"]
        ACTIVITY_TYPEs = ["GPRS", "DATA_SESSION"]
    else:
        USAGE_TYPEs = ["MO", "MT", "MOC", "MTC", "SMS"]
        ACTIVITY_TYPEs = ["VOICE_CALL", "SMS", "VIDEO_CALL"]
    
    record.append(random.choice(USAGE_TYPEs))
    record.append(random.choice(ACTIVITY_TYPEs))
    record.append(subscriber_msisdn)
    
    if device_tracker:
        if allow_swap:
            device_info = device_tracker.get_or_assign_device(
                subscriber_msisdn, 
                allow_swap=True, 
                swap_time=swap_time,
                segment=segment,
                subscription_date=subscription_date
            )
        else:
            device_info = device_tracker.get_or_assign_device(
                subscriber_msisdn, 
                allow_swap=False,
                segment=segment,
                subscription_date=subscription_date
            )
        
        record.append(device_info.imsi)
        record.append(device_info.imei)
        record.append(device_info.device_name)
        record.append(device_info.device_brand)
        
        subscription_date = device_info.subscription_date
    else:
        device = DEVICE_DB.get_random_device()
        imei = generate_random_imei(device["tac"])
        record.append(generate_random_imsi())
        record.append(imei)
        record.append(device["name"])
        record.append(device["brand"])
        subscription_date = random_subscription_datetime()
    
    record.append(tower["tower_id"])
    record.append(tower["name"])
    record.append(tower["lac"])
    record.append(tower["cell_id"])
    record.append(tower["latitude"])
    record.append(tower["longitude"])
    record.append(tower["tower_type"])
    record.append(tower["environment"])
    
    days_back = random.randint(0, Config.GENERATION_DAYS_BACK)
    start_time = random_temporal_datetime(days_back, segment, temporal_patterns)
    if subscription_date and start_time < subscription_date:
        span_seconds = max(0, int((datetime.datetime.now() - subscription_date).total_seconds()))
        start_time = subscription_date + datetime.timedelta(seconds=random.randint(0, span_seconds)) if span_seconds else subscription_date
    record.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    if is_gprs:
        duration = random.randint(60, 7200)
    else:
        if segment == SubscriberSegment.HIGH_VALUE:
            duration = random.randint(60, 1200)
        elif segment == SubscriberSegment.BUSINESS:
            duration = random.randint(30, 900)
        else:
            duration = random.randint(1, 600)
    
    end_time = start_time + datetime.timedelta(seconds=duration)
    record.append(end_time.strftime("%Y-%m-%d %H:%M:%S"))
    record.append(duration)
    
    if social_graph and subscriber_msisdn:
        contact = social_graph.get_weighted_contact(subscriber_msisdn)
        if contact and random.random() < 0.7:
            CALLED_MSISDN = contact
        else:
            CALLED_MSISDN = random_msisdn()
    else:
        CALLED_MSISDN = random_msisdn()
        while CALLED_MSISDN == subscriber_msisdn:
            CALLED_MSISDN = random_msisdn()
    
    if is_gprs:
        record.append(random.choice(["internet", "wap", "mms", "ims"]))
        record.append(record[18])
    else:
        record.append(CALLED_MSISDN)
        record.append(CALLED_MSISDN)
    
    data_volume = random_data_volume() if is_gprs else 0.0
    record.append(data_volume)
    
    if is_gprs:
        charge = random_charge(duration, "GPRS", data_volume)
    else:
        charge = random_charge(duration, record[0])
    record.append(charge)
    
    if device_tracker and subscriber_msisdn in device_tracker.subscriber_devices:
        device = device_tracker.subscriber_devices[subscriber_msisdn]
        subscription_date = device.subscription_date
        sim_change_date = device.sim_change_date
    else:
        subscription_date = random_subscription_datetime()
        sim_change_date = random_sim_change_datetime(subscription_date)
    
    record.append(subscription_date.strftime("%Y-%m-%d %H:%M:%S"))
    record.append(sim_change_date.strftime("%Y-%m-%d %H:%M:%S"))
    
    record.append("")  # Fraud Type (index 24)
    record.append("")  # Sophistication (index 25)
    record.append("")  # Confidence (index 26)
    record.append("")  # FRAUD_REASON (index 27)
    
    return record

def generate_random_imsi() -> str:
    mcc = random.choice(["655", "310", "311", "312"])
    if mcc == "655":
        mnc = random.choice(["01", "02", "03"])
    else:
        mnc = random.choice(["010", "020", "030"])
    msn_length = 15 - len(mcc) - len(mnc)
    msn = ''.join([str(random.randint(0, 9)) for _ in range(msn_length)])
    return f"{mcc}{mnc}{msn}"

def generate_random_imei(tac: str = None) -> str:
    if tac is None:
        device = DEVICE_DB.get_random_device()
        tac = device["tac"]
    snr = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    imei_without_check = f"{tac}{snr}"
    check_digit = calculate_luhn_check_digit(imei_without_check)
    return f"{tac}{snr}{check_digit}"

def calculate_luhn_check_digit(number: str) -> int:
    digits = [int(d) for d in number]
    for i in range(len(digits) - 1, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] = digits[i] - 9
    total = sum(digits)
    check_digit = (10 - (total % 10)) % 10
    return check_digit

# ===================================================================
# ENHANCED SIM SWAP FRAUD PATTERN GENERATOR
# ===================================================================

class EnhancedSIMSwapPatternGenerator:
    """Enhanced SIM swap fraud with two distinct devices and proper tracking"""
    
    @staticmethod
    def generate_sim_swap_scenario(legitimate_records: List, 
                                  sophistication: FraudSophistication,
                                  device_tracker: EnhancedDeviceAssignmentTracker,
                                  social_graph: SocialGraphManager) -> List:
        
        fraud_records = []
        
        if not legitimate_records:
            msisdn = random_msisdn()
            num_legit = random.randint(5, 10)
            legitimate_records = []
            
            subscription_date = random_subscription_datetime()
            for _ in range(num_legit):
                rec = generate_base_cdr(
                    msisdn, 
                    device_tracker, 
                    allow_swap=False,
                    subscription_date=subscription_date
                )
                legitimate_records.append(rec)
        
        msisdn = legitimate_records[0][2]
        
        if msisdn in device_tracker.subscriber_devices:
            old_device = device_tracker.subscriber_devices[msisdn]
            old_imei = old_device.imei
            old_imsi = old_device.imsi
            old_device_name = old_device.device_name
            old_device_brand = old_device.device_brand
        else:
            old_imei = legitimate_records[0][4] if len(legitimate_records[0]) > 4 else "UNKNOWN"
            old_imsi = legitimate_records[0][3] if len(legitimate_records[0]) > 3 else "UNKNOWN"
            old_device_name = legitimate_records[0][5] if len(legitimate_records[0]) > 5 else "UNKNOWN"
            old_device_brand = legitimate_records[0][6] if len(legitimate_records[0]) > 6 else "UNKNOWN"
        
        durations = []
        for r in legitimate_records:
            try:
                durations.append(int(r[17]))
            except:
                pass
        avg_duration = np.mean(durations) if durations else 120
        
        last_legit_time = datetime.datetime.strptime(legitimate_records[-1][15], "%Y-%m-%d %H:%M:%S")
        swap_time = min(
            last_legit_time + datetime.timedelta(hours=random.randint(1, 48)),
            datetime.datetime.now()
        )
        
        subscription_date = datetime.datetime.strptime(
            legitimate_records[-1][22], "%Y-%m-%d %H:%M:%S"
        ) if len(legitimate_records[-1]) > 22 else random_subscription_datetime()
        
        swap_record = generate_base_cdr(
            msisdn, 
            device_tracker, 
            allow_swap=True, 
            swap_time=swap_time,
            subscription_date=subscription_date
        )
        
        if msisdn in device_tracker.subscriber_devices:
            new_device = device_tracker.subscriber_devices[msisdn]
            new_imei = new_device.imei
            new_imsi = new_device.imsi
            new_device_name = new_device.device_name
            new_device_brand = new_device.device_brand
            sim_change_date = new_device.sim_change_date
        else:
            new_imei = swap_record[4] if len(swap_record) > 4 else "UNKNOWN"
            new_imsi = swap_record[3] if len(swap_record) > 3 else "UNKNOWN"
            new_device_name = swap_record[5] if len(swap_record) > 5 else "UNKNOWN"
            new_device_brand = swap_record[6] if len(swap_record) > 6 else "UNKNOWN"
            sim_change_date = swap_time
        
        swap_record[0] = "SMS"
        swap_record[1] = "SMS"
        swap_record[17] = 0
        swap_record[21] = 0.01
        swap_record[24] = "SIM_SWAP_EVENT"
        swap_record[25] = sophistication.name
        swap_record[26] = 0.99
        swap_record[27] = (
            f"SIM SWAP: {old_device_name} -> {new_device_name} | "
            f"IMEI: {old_imei[:8]}... -> {new_imei[:8]}... | "
            f"IMSI: {old_imsi[:6]}... -> {new_imsi[:6]}... | "
            f"Swap: {swap_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        fraud_records.append(swap_record)
        
        num_records = random.randint(8, 20)
        
        for i in range(num_records):
            record = generate_base_cdr(
                msisdn, 
                device_tracker, 
                allow_swap=False,
                subscription_date=subscription_date
            )
            
            if sophistication == FraudSophistication.NAIVE:
                record[0] = "MOC"
                record[1] = "VOICE_CALL"
                record[17] = random.randint(600, 3600)
                if random.random() < 0.9:
                    record[18] = f"+44{random.randint(7000000000, 7999999999)}"
                else:
                    record[18] = f"+1{random.randint(8000000000, 8999999999)}"
                record[19] = record[18]
                record[21] = random_charge(record[17], record[0])
                
            elif sophistication == FraudSophistication.MODERATE:
                record[0] = random.choice(["MOC", "MO", "SMS"])
                record[1] = "VOICE_CALL" if record[0] in ["MOC", "MO"] else "SMS"
                record[17] = random.randint(120, 1200) if record[0] != "SMS" else 0
                if random.random() < 0.5:
                    record[18] = f"+44{random.randint(7000000000, 7999999999)}"
                else:
                    record[18] = random_msisdn()
                record[19] = record[18]
                record[21] = random_charge(record[17], record[0])
                
            else:
                record[0] = random.choice(["MO", "MT", "MOC", "MTC", "SMS"])
                record[1] = "VOICE_CALL" if record[0] in ["MO", "MT", "MOC", "MTC"] else "SMS"
                record[17] = random.randint(30, 600) if record[0] != "SMS" else 0
                
                rand_val = random.random()
                if rand_val < 0.1:
                    record[18] = f"+900{random.randint(100000, 999999)}"
                elif rand_val < 0.2:
                    record[18] = f"+44{random.randint(7000000000, 7999999999)}"
                else:
                    contacts = social_graph.get_contacts(msisdn)
                    if contacts and random.random() < 0.7:
                        record[18] = random.choice(contacts)
                    else:
                        record[18] = random_msisdn()
                record[19] = record[18]
                record[21] = random_charge(record[17], record[0])
            
            available_seconds = max(0, int((datetime.datetime.now() - swap_time).total_seconds()))
            fraud_time = swap_time + datetime.timedelta(
                seconds=random.randint(0, available_seconds) if available_seconds else 0
            )
            record[15] = fraud_time.strftime("%Y-%m-%d %H:%M:%S")
            
            if record[17] > 0:
                start = datetime.datetime.strptime(record[15], "%Y-%m-%d %H:%M:%S")
                record[16] = (start + datetime.timedelta(seconds=record[17])).strftime("%Y-%m-%d %H:%M:%S")
            else:
                record[16] = record[15]
            
            record[24] = "SIM_SWAP"
            record[25] = sophistication.name
            record[26] = 0.90 if sophistication == FraudSophistication.NAIVE else \
                         0.85 if sophistication == FraudSophistication.MODERATE else 0.80
            record[26] += 0.10 * random.random()
            
            record[27] = (
                f"SIM SWAP FRAUD | "
                f"Device: {new_device_name} | "
                f"Duration: {record[17]}s | "
                f"Dest: {record[18]} | "
                f"Charge: R{record[21]:.2f}"
            )
            
            fraud_records.append(record)
        
        return fraud_records

# ===================================================================
# ENHANCED SUBSCRIPTION FRAUD PATTERN GENERATOR
# ===================================================================

class EnhancedSubscriptionFraudPatternGenerator:
    """Enhanced subscription fraud with social network integration"""
    
    @staticmethod
    def generate_subscription_fraud_pattern(sophistication: FraudSophistication,
                                          device_tracker: EnhancedDeviceAssignmentTracker,
                                          social_graph: SocialGraphManager) -> List:
        """Generate one coherent new-account fraud case with scenario-level SIM changes."""
        fraud_records = []
        msisdn = random_msisdn()
        num_records = random.randint(10, 25)
        now = datetime.datetime.now()
        subscription_date = now - datetime.timedelta(days=random.randint(1, 7))

        device_tracker.get_or_assign_device(
            msisdn, allow_swap=False, segment=SubscriberSegment.NEW,
            subscription_date=subscription_date
        )

        # SIM changes are case events, not something repeated for every CDR row.
        sim_change_count = 0
        if sophistication in [FraudSophistication.MODERATE, FraudSophistication.ADVANCED, FraudSophistication.EVOLVING]:
            sim_change_count = random.choices([0, 1, 2], weights=[0.45, 0.45, 0.10], k=1)[0]

        last_sim_change = subscription_date
        if sim_change_count:
            lifetime_seconds = max(1, int((now - subscription_date).total_seconds()))
            offsets = sorted(random.sample(range(1, lifetime_seconds + 1), k=min(sim_change_count, lifetime_seconds)))
            for offset in offsets:
                swap_time = subscription_date + datetime.timedelta(seconds=offset)
                device_tracker.get_or_assign_device(
                    msisdn, allow_swap=True, swap_time=swap_time, subscription_date=subscription_date
                )
                last_sim_change = swap_time

        for _ in range(num_records):
            record = generate_base_cdr(
                msisdn, device_tracker, allow_swap=False,
                segment=SubscriberSegment.NEW, temporal_patterns=None,
                social_graph=social_graph, subscription_date=subscription_date
            )

            record[0] = random.choice(["MOC", "MO", "GPRS", "DATA"])
            record[1] = "DATA_SESSION" if record[0] in ["GPRS", "DATA"] else "VOICE_CALL"

            event_lower = max(subscription_date, last_sim_change)
            available_seconds = max(0, int((now - event_lower).total_seconds()))
            event_time = event_lower + datetime.timedelta(
                seconds=random.randint(0, available_seconds) if available_seconds else 0
            )
            record[15] = event_time.strftime("%Y-%m-%d %H:%M:%S")

            if record[0] in ["GPRS", "DATA"]:
                record[20] = random_data_volume() * random.uniform(1.5, 5.0)
                record[17] = random.randint(300, 7200)
                record[21] = random_charge(record[17], "GPRS", record[20])
            else:
                record[17] = random.randint(60, 2400)
                intl_prob = 0.9 if sophistication == FraudSophistication.NAIVE else \
                            0.6 if sophistication == FraudSophistication.MODERATE else 0.3
                record[18] = f"+44{random.randint(7000000000, 7999999999)}" if random.random() < intl_prob else random_msisdn()
                record[19] = record[18]
                record[21] = random_charge(record[17], record[0])

            record[16] = (event_time + datetime.timedelta(seconds=record[17])).strftime("%Y-%m-%d %H:%M:%S")
            record[22] = subscription_date.strftime("%Y-%m-%d %H:%M:%S")
            record[23] = last_sim_change.strftime("%Y-%m-%d %H:%M:%S")
            record[24] = "SUBSCRIPTION_FRAUD"
            record[25] = sophistication.name
            record[26] = 0.80 + 0.20 * random.random()
            record[27] = (
                f"SUBSCRIPTION FRAUD | New: {subscription_date.strftime('%Y-%m-%d')} | "
                f"SIM changes: {sim_change_count} | Duration: {record[17]}s | Charge: R{record[21]:.2f}"
            )
            fraud_records.append(record)

        return fraud_records

class CallForwardingFraudGenerator:
    """Generates call forwarding fraud patterns"""
    
    @staticmethod
    def generate_fraud(sophistication: FraudSophistication,
                      device_tracker: EnhancedDeviceAssignmentTracker) -> List:
        
        fraud_records = []
        msisdn = random_msisdn()
        subscription_date = random_subscription_datetime()
        
        for _ in range(random.randint(5, 10)):
            record = generate_base_cdr(
                msisdn, 
                device_tracker, 
                allow_swap=False,
                subscription_date=subscription_date
            )
            record[24] = "LEGITIMATE"
            record[25] = "N/A"
            record[26] = 0.0
            record[27] = "Normal activity - no fraud detected"
            fraud_records.append(record)
        
        forwarding_time = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 3))
        forwarding_number = f"+900{random.randint(100000, 999999)}"
        
        activation_record = generate_base_cdr(
            msisdn, 
            device_tracker, 
            allow_swap=False,
            subscription_date=subscription_date
        )
        activation_record[0] = "SMS"
        activation_record[1] = "SMS"
        activation_record[15] = forwarding_time.strftime("%Y-%m-%d %H:%M:%S")
        activation_record[18] = "SERVICE"
        activation_record[19] = f"FORWARD_{forwarding_number}"
        activation_record[17] = 0
        activation_record[24] = "CALL_FORWARDING_FRAUD"
        activation_record[25] = sophistication.name
        activation_record[26] = 0.85 + 0.10 * random.random()
        activation_record[27] = f"Forwarding to {forwarding_number}"
        fraud_records.append(activation_record)
        
        num_fraud = random.randint(8, 15)
        for i in range(num_fraud):
            record = generate_base_cdr(
                msisdn, 
                device_tracker, 
                allow_swap=False,
                subscription_date=subscription_date
            )
            
            record[0] = "MTC"
            record[1] = "FORWARDED_CALL"
            record[17] = random.randint(60, 1800)
            record[18] = random_msisdn()
            record[19] = forwarding_number
            record[21] = random_charge(record[17], "MTC") * 1.5
            
            call_time = min(
                forwarding_time + datetime.timedelta(hours=random.randint(2, 48)),
                datetime.datetime.now()
            )
            record[15] = call_time.strftime("%Y-%m-%d %H:%M:%S")
            record[16] = (call_time + datetime.timedelta(seconds=record[17])).strftime("%Y-%m-%d %H:%M:%S")
            
            record[24] = "CALL_FORWARDING_FRAUD"
            record[25] = sophistication.name
            record[26] = 0.75 + 0.20 * random.random()
            record[27] = f"Forwarded to {forwarding_number} | Duration: {record[17]}s | R{record[21]:.2f}"
            
            fraud_records.append(record)
        
        return fraud_records

class PremiumSMSFraudGenerator:
    """Generates premium SMS fraud patterns"""
    
    @staticmethod
    def generate_fraud(sophistication: FraudSophistication,
                      device_tracker: EnhancedDeviceAssignmentTracker) -> List:
        
        fraud_records = []
        msisdn = random_msisdn()
        subscription_date = random_subscription_datetime()
        
        for _ in range(random.randint(3, 8)):
            record = generate_base_cdr(
                msisdn, 
                device_tracker, 
                allow_swap=False,
                subscription_date=subscription_date
            )
            record[24] = "LEGITIMATE"
            record[25] = "N/A"
            record[26] = 0.0
            record[27] = "Normal activity - no fraud detected"
            fraud_records.append(record)
        
        num_sms = random.randint(15, 40)
        premium_shortcodes = ["44555", "44666", "44777", "44888", "44999"]
        
        for i in range(num_sms):
            record = generate_base_cdr(
                msisdn, 
                device_tracker, 
                allow_swap=False,
                subscription_date=subscription_date
            )
            record[0] = "SMS"
            record[1] = "SMS"
            record[17] = 0
            record[18] = random.choice(premium_shortcodes)
            record[19] = record[18]
            record[21] = random.uniform(1.0, 5.0)
            
            sms_time = datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 72))
            record[15] = sms_time.strftime("%Y-%m-%d %H:%M:%S")
            record[16] = record[15]
            
            record[24] = "PREMIUM_SMS_FRAUD"
            record[25] = sophistication.name
            record[26] = 0.85 + 0.10 * random.random()
            record[27] = f"Premium SMS to {record[18]} | R{record[21]:.2f}"
            
            fraud_records.append(record)
        
        return fraud_records

class DataRoamingFraudGenerator:
    """Generates data roaming fraud patterns"""
    
    @staticmethod
    def generate_fraud(sophistication: FraudSophistication,
                      device_tracker: EnhancedDeviceAssignmentTracker) -> List:
        
        fraud_records = []
        msisdn = random_msisdn()
        num_records = random.randint(8, 15)
        roaming_tower = TOWER_DB.get_roaming_tower()
        subscription_date = random_subscription_datetime()
        
        for i in range(num_records):
            record = generate_base_cdr(
                msisdn, 
                device_tracker, 
                allow_swap=False,
                subscription_date=subscription_date
            )
            
            record[0] = "GPRS"
            record[1] = "DATA_SESSION"
            record[20] = random_data_volume() * random.uniform(2.0, 10.0)
            
            record[7] = roaming_tower["tower_id"]
            record[8] = roaming_tower["name"]
            record[9] = roaming_tower["lac"]
            record[10] = roaming_tower["cell_id"]
            record[11] = roaming_tower["latitude"]
            record[12] = roaming_tower["longitude"]
            record[13] = "ROAMING"
            record[14] = "ROAMING"
            
            record[17] = random.randint(600, 7200)
            record[21] = random_charge(record[17], "GPRS", record[20]) * 3.0
            
            call_time = datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 48))
            record[15] = call_time.strftime("%Y-%m-%d %H:%M:%S")
            record[16] = (call_time + datetime.timedelta(seconds=record[17])).strftime("%Y-%m-%d %H:%M:%S")
            
            record[24] = "DATA_ROAMING_FRAUD"
            record[25] = sophistication.name
            record[26] = 0.80 + 0.15 * random.random()
            record[27] = f"Roaming: {record[20]:.1f}MB | R{record[21]:.2f} | {roaming_tower['name']}"
            
            fraud_records.append(record)
        
        return fraud_records

# ===================================================================
# ENHANCED MAIN GENERATOR
# ===================================================================

class EnhancedReinforcementLearningCDRGenerator:
    """Enhanced CDR generator with all improvements"""
    
    def __init__(self, num_records: int = Config.NUM_RECORDS, 
                 fraud_ratio: float = Config.FRAUD_RATIO,
                 config: Config = None):
        self.num_records = num_records
        self.fraud_ratio = fraud_ratio
        self.num_fraud_records = int(num_records * fraud_ratio)
        self.num_legitimate_records = num_records - self.num_fraud_records
        self.config = config or Config()
        
        self.device_tracker = EnhancedDeviceAssignmentTracker()
        self.temporal_patterns = TemporalPatternGenerator()
        self.social_graph = SocialGraphManager()
        self.population: List[Dict[str, Any]] = []
        self._fraud_case_counter = 0

        # Reproducibility must cover both Python and NumPy RNGs.
        random.seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)

        self.generation_time = 0
    
    def generate(self, output_file: str = Config.OUTPUT_FILE) -> List:
        """Main generation function"""
        
        start_time = time.time()
        logger.info(f"Generating {self.num_records} CDR records...")
        logger.info(f"  - Legitimate: {self.num_legitimate_records}")
        logger.info(f"  - Fraudulent: {self.num_fraud_records}")
        
        all_records = []
        
        logger.info("Building subscriber population and social network...")
        num_subscribers = min(5000, max(100, self.num_records // 20))
        self._initialize_population(num_subscribers)
        self.social_graph.build_social_network([p["msisdn"] for p in self.population])
        
        logger.info("Generating legitimate records...")
        legitimate_records = self._generate_legitimate_records(self.num_legitimate_records)
        all_records.extend(legitimate_records)
        
        logger.info("Generating fraudulent records...")
        fraud_records = self._generate_fraud_records(self.num_fraud_records, legitimate_records)
        all_records.extend(fraud_records)

        # Scenario generators may override start time and/or duration. Recompute
        # end time centrally so every exported row is temporally consistent.
        for record in all_records:
            if len(record) > 17 and record[15]:
                start_dt = datetime.datetime.strptime(record[15], "%Y-%m-%d %H:%M:%S")
                duration_sec = int(float(record[17])) if record[17] else 0
                record[16] = (start_dt + datetime.timedelta(seconds=max(0, duration_sec))).strftime("%Y-%m-%d %H:%M:%S")
        
        if Config.PRESERVE_CHRONOLOGY:
            logger.info("Sorting records chronologically...")
            all_records.sort(key=lambda r: r[15] if len(r) > 15 else "")
        else:
            logger.info("Shuffling records...")
            random.shuffle(all_records)

        # Defensive guard: generation must honor the requested row count exactly.
        all_records = all_records[:self.num_records]

        # Add customer name and address from subscriber profiles
        logger.info("Adding customer name and address...")
        for record in all_records:
            msisdn = record[2] if len(record) > 2 else ""
            if msisdn in self.device_tracker.subscriber_profiles:
                profile = self.device_tracker.subscriber_profiles[msisdn]
                record.append(profile.full_name)
                record.append(profile.address)
            else:
                record.append("UNKNOWN")
                record.append("UNKNOWN")

        if Config.ENABLE_VALIDATION:
            self._validate_records(all_records)

        logger.info(f"Writing to {output_file}...")
        self._write_csv(all_records, output_file)
        
        self.generation_time = time.time() - start_time
        self._print_statistics(all_records)
        
        return all_records
    
    def _initialize_population(self, num_subscribers: int):
        """Create one coherent subscriber population reused by profiles, devices and social graph."""
        self.population = []
        seen = set()
        segment_weights = [
            (SubscriberSegment.HIGH_VALUE, 0.10),
            (SubscriberSegment.BUSINESS, 0.15),
            (SubscriberSegment.AVERAGE, 0.50),
            (SubscriberSegment.BUDGET, 0.25),
        ]
        segments, weights = zip(*segment_weights)

        # Get a random tower to base addresses on
        random_tower = TOWER_DB.get_random_tower()

        while len(self.population) < num_subscribers:
            msisdn = random_msisdn()
            if msisdn in seen:
                continue
            seen.add(msisdn)
            segment = random.choices(segments, weights=weights, k=1)[0]
            subscription_date = random_subscription_datetime()
            device_info = self.device_tracker.get_or_assign_device(
                msisdn, allow_swap=False, segment=segment, subscription_date=subscription_date
            )

            # Generate name and address using the random tower's name
            tower_name = random_tower["name"]
            full_name = generate_customer_name()
            address = generate_customer_address(tower_name)

            # Assign to profile
            profile = self.device_tracker.subscriber_profiles[msisdn]
            profile.full_name = full_name
            profile.address = address

            self.population.append({
                "msisdn": msisdn,
                "segment": segment,
                "subscription_date": subscription_date,
            })

    def _generate_legitimate_records(self, count: int) -> List:
        """Generate legitimate records from the shared subscriber population."""
        if count == 0:
            return []

        if not self.population:
            self._initialize_population(max(1, min(5000, count // 15)))

        records = []
        for i in range(count):
            person = random.choice(self.population)
            subscriber = person["msisdn"]
            segment = person["segment"]
            subscription_date = person["subscription_date"]

            # Legitimate device upgrades are handled by the tracker. Do not misuse
            # fraudulent SIM-swap logic for normal customer device changes.
            record = generate_base_cdr(
                subscriber,
                self.device_tracker,
                allow_swap=False,
                segment=segment,
                temporal_patterns=self.temporal_patterns,
                social_graph=self.social_graph,
                subscription_date=subscription_date,
            )

            record[24] = "LEGITIMATE"
            record[25] = "N/A"
            record[26] = 0.0
            record[27] = "Normal activity - no fraud detected"
            record.extend([0, "", "", "", "SYNTHETIC_GROUND_TRUTH"])
            records.append(record)

            if i > 0 and i % 10000 == 0:
                logger.info(f"  Generated {i}/{count} legitimate records...")

        return records

    def _generate_fraud_records(self, count: int, legitimate_records: List) -> List:
        """Generate exactly ``count`` fraudulent CDR rows, grouped into fraud cases."""
        if count == 0:
            return []

        probabilities = {
            "SIM_SWAP": Config.SIM_SWAP_PROBABILITY,
            "SUBSCRIPTION_FRAUD": Config.SUBSCRIPTION_FRAUD_PROBABILITY,
            "CALL_FORWARDING": Config.CALL_FORWARDING_FRAUD_PROBABILITY,
            "PREMIUM_SMS": Config.PREMIUM_SMS_FRAUD_PROBABILITY,
            "DATA_ROAMING": Config.DATA_ROAMING_FRAUD_PROBABILITY,
        }
        fraud_types = {k: int(count * p) for k, p in probabilities.items()}
        fraud_types["SIM_SWAP"] += count - sum(fraud_types.values())

        legit_by_subscriber = defaultdict(list)
        for row in legitimate_records:
            if len(row) > 2:
                legit_by_subscriber[row[2]].append(row)
        for rows in legit_by_subscriber.values():
            rows.sort(key=lambda r: r[15])

        fraud_records = []
        for fraud_type, target_rows in fraud_types.items():
            logger.info(f"  Generating {target_rows} {fraud_type} rows...")
            type_rows = []
            attempts = 0
            while len(type_rows) < target_rows:
                attempts += 1
                if attempts > max(100, target_rows * 5):
                    raise RuntimeError(f"Unable to generate enough {fraud_type} rows")

                soph = self._get_sophistication_level()
                if fraud_type == "SIM_SWAP":
                    eligible = [m for m, rows in legit_by_subscriber.items() if rows]
                    if eligible:
                        msisdn = random.choice(eligible)
                        legit_base = legit_by_subscriber[msisdn][-min(len(legit_by_subscriber[msisdn]), random.randint(3, 8)):]
                    else:
                        legit_base = []
                    generated = EnhancedSIMSwapPatternGenerator.generate_sim_swap_scenario(
                        legit_base, soph, self.device_tracker, self.social_graph
                    )
                elif fraud_type == "SUBSCRIPTION_FRAUD":
                    generated = EnhancedSubscriptionFraudPatternGenerator.generate_subscription_fraud_pattern(
                        soph, self.device_tracker, self.social_graph
                    )
                elif fraud_type == "CALL_FORWARDING":
                    generated = CallForwardingFraudGenerator.generate_fraud(soph, self.device_tracker)
                elif fraud_type == "PREMIUM_SMS":
                    generated = PremiumSMSFraudGenerator.generate_fraud(soph, self.device_tracker)
                elif fraud_type == "DATA_ROAMING":
                    generated = DataRoamingFraudGenerator.generate_fraud(soph, self.device_tracker)
                else:
                    continue

                # Some scenario generators include legitimate warm-up rows.
                case_rows = [r for r in generated if len(r) > 24 and r[24] != "LEGITIMATE"]
                if not case_rows:
                    continue

                self._fraud_case_counter += 1
                case_id = f"CASE-{self._fraud_case_counter:08d}"
                times = [r[15] for r in case_rows if len(r) > 15 and r[15]]
                case_start = min(times) if times else ""
                case_end = max(times) if times else ""

                for r in case_rows:
                    r.extend([1, case_id, case_start, case_end, "SYNTHETIC_GROUND_TRUTH"])
                    type_rows.append(r)
                    if len(type_rows) >= target_rows:
                        break

            fraud_records.extend(type_rows[:target_rows])

        if len(fraud_records) != count:
            raise AssertionError(f"Expected {count} fraud rows, generated {len(fraud_records)}")
        return fraud_records

    def _get_sophistication_level(self) -> FraudSophistication:
        r = random.random()
        if r < 0.25:
            return FraudSophistication.NAIVE
        elif r < 0.60:
            return FraudSophistication.MODERATE
        elif r < 0.85:
            return FraudSophistication.ADVANCED
        else:
            return FraudSophistication.EVOLVING
    
    def _write_csv(self, records: List, output_file: str):
        headers = [
            "USAGE_TYPE", "ACTIVITY_TYPE", "CALLING_MSISDN", "IMSI", "IMEI",
            "DEVICE_NAME", "DEVICE_BRAND",
            "TOWER_ID", "TOWER_NAME", "LAC", "CELL_ID", 
            "LATITUDE", "LONGITUDE", "TOWER_TYPE", "ENVIRONMENT",
            "CALL_START_DATE", "CALL_END_DATE", "CALL_DURATION_SEC", 
            "CALLED_MSISDN", "DIALED_NUMBER", "DATA_VOLUME_MB", "AMOUNT_CHARGE_FEE",
            "SUBSCRIPTION_DATE", "LAST_SIM_CHANGE_DATE", 
            "FRAUD_TYPE", "FRAUD_SOPHISTICATION", "FRAUD_CONFIDENCE",
            "FRAUD_REASON", "IS_FRAUD", "FRAUD_CASE_ID", "FRAUD_START_TIME",
            "FRAUD_END_TIME", "LABEL_SOURCE", "CUSTOMER_NAME", "ADDRESS"
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(records)
    
    def _validate_records(self, records: List):
        """Validate schema, counts, labels and basic temporal consistency."""
        if len(records) != self.num_records:
            raise AssertionError(f"Expected {self.num_records} rows, got {len(records)}")

        # Now expecting 35 columns (33 original + 2 new)
        bad_width = [i for i, r in enumerate(records) if len(r) != 35]
        if bad_width:
            raise AssertionError(f"Rows with invalid schema width: {bad_width[:5]}")

        fraud_count = sum(int(r[28]) for r in records)
        expected_fraud = self.num_fraud_records
        if fraud_count != expected_fraud:
            raise AssertionError(f"Expected {expected_fraud} fraud rows, got {fraud_count}")

        for i, r in enumerate(records):
            try:
                start = datetime.datetime.strptime(r[15], "%Y-%m-%d %H:%M:%S")
                end = datetime.datetime.strptime(r[16], "%Y-%m-%d %H:%M:%S")
                if end < start:
                    raise AssertionError(f"Row {i}: end time precedes start time")
                if start > datetime.datetime.now():
                    raise AssertionError(f"Row {i}: start time is in the future")
            except ValueError as exc:
                raise AssertionError(f"Row {i}: invalid datetime: {exc}") from exc

        logger.info("Validation passed: exact row count, schema width, fraud ratio and timestamps")

    def _print_statistics(self, records: List):
        fraud_types = defaultdict(int)
        sophistication_levels = defaultdict(int)
        sim_swap_count = defaultdict(int)
        
        for record in records:
            if len(record) >= 27:
                fraud_type = record[24]
                soph = record[25]
                fraud_types[fraud_type] += 1
                sophistication_levels[soph] += 1
        
        for msisdn, count in self.device_tracker.sim_swap_count.items():
            sim_swap_count[count] += 1
        
        logger.info("\n--- Generation Statistics ---")
        logger.info(f"Total records: {len(records)}")
        logger.info(f"Generation time: {self.generation_time:.2f} seconds")
        logger.info(f"Speed: {len(records) / self.generation_time:.0f} records/second")
        
        logger.info("\nFraud Type Distribution:")
        for fraud_type, count in sorted(fraud_types.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(records) * 100
            logger.info(f"  {fraud_type}: {count} ({pct:.2f}%)")
        
        logger.info("\nSophistication Distribution:")
        for level, count in sorted(sophistication_levels.items(), key=lambda x: x[1], reverse=True):
            if level != 'N/A':
                pct = count / len(records) * 100
                logger.info(f"  {level}: {count} ({pct:.2f}%)")
        
        logger.info("\nSIM Swap Statistics:")
        logger.info(f"  Total SIM Swap Events: {len(self.device_tracker.sim_swap_events)}")
        logger.info(f"  Unique Subscribers with Swaps: {len(self.device_tracker.sim_swap_count)}")
        
        for swap_count, num_subscribers in sorted(sim_swap_count.items()):
            logger.info(f"  {swap_count} swaps: {num_subscribers} subscribers")
        
        if self.device_tracker.sim_swap_events:
            logger.info("\nSample SIM Swap Events:")
            for event in self.device_tracker.sim_swap_events[:3]:
                logger.info(f"  {event['msisdn']}: {event['old_device_name']} -> {event['new_device_name']} "
                          f"(IMEI: {event['old_imei'][:8]}... -> {event['new_imei'][:8]}...) "
                          f"at {event['swap_time']}")

# ===================================================================
# NOT NECCESSARY - Q-TELEGUARD WILL HANDLE THIS PART
# ===================================================================

class EnhancedRLEnvironment:
    """Enhanced RL environment with advanced reward function"""
    
    def __init__(self, generator: EnhancedReinforcementLearningCDRGenerator):
        self.generator = generator
        self.records = []
        self.current_step = 0
        self.total_steps = 0
        self.subscriber_state = defaultdict(dict)
        self.detection_history = defaultdict(list)
        self.cumulative_reward = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    
    def load_records(self, file_path: str):
        self.records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                self.records.append(row)
        self.total_steps = len(self.records)
        logger.info(f"Loaded {self.total_steps} records from {file_path}")
    
    def reset(self):
        self.current_step = 0
        self.subscriber_state = defaultdict(dict)
        self.detection_history = defaultdict(list)
        self.cumulative_reward = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        if not Config.PRESERVE_CHRONOLOGY:
            random.shuffle(self.records)
        return self._get_initial_state()
    
    def step(self, action: int):
        if self.current_step >= len(self.records):
            return None, 0, True, {}
        
        record = self.records[self.current_step]
        msisdn = record[2] if len(record) > 2 else "UNKNOWN"
        
        fraud_type = record[24] if len(record) > 24 else "LEGITIMATE"
        is_fraud = bool(int(record[28])) if len(record) > 28 and str(record[28]).strip() else fraud_type != "LEGITIMATE"
        sophistication = record[25] if len(record) > 25 else "N/A"
        confidence = float(record[26]) if len(record) > 26 and record[26] else 0.0
        fraud_reason = record[27] if len(record) > 27 else ""
        
        reward = self._calculate_advanced_reward(
            action, is_fraud, sophistication, confidence, msisdn
        )
        
        self._update_subscriber_state(msisdn, record, action)
        
        self.cumulative_reward += reward
        self.total_predictions += 1
        
        if action == 1 and is_fraud:
            self.confusion_matrix["tp"] += 1
            self.correct_predictions += 1
        elif action == 0 and not is_fraud:
            self.confusion_matrix["tn"] += 1
            self.correct_predictions += 1
        elif action == 1 and not is_fraud:
            self.confusion_matrix["fp"] += 1
        elif action == 0 and is_fraud:
            self.confusion_matrix["fn"] += 1
        
        self.current_step += 1
        done = (self.current_step >= len(self.records))
        
        features = self._extract_features(record, msisdn)
        
        info = {
            "fraud_type": fraud_type,
            "sophistication": sophistication,
            "confidence": confidence,
            "fraud_reason": fraud_reason,
            "cumulative_reward": self.cumulative_reward,
            "accuracy": self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        }
        
        return features, reward, done, info
    
    def _calculate_advanced_reward(self, action: int, is_fraud: bool, 
                                  sophistication: str, confidence: float,
                                  msisdn: str) -> float:
        reward = 0
        
        if action == is_fraud:
            reward = 1.0
        else:
            reward = -1.0
        
        if is_fraud:
            if sophistication == "ADVANCED" and action == 1:
                reward += 0.5
            elif sophistication == "MODERATE" and action == 1:
                reward += 0.3
            elif sophistication == "NAIVE" and action == 1:
                reward += 0.1
            
            if action == 1 and confidence > 0.9:
                reward += 0.3
            
            if action == 0:
                if sophistication == "ADVANCED":
                    reward -= 1.0
                elif sophistication == "MODERATE":
                    reward -= 0.5
        
        return max(-2.0, min(2.0, reward))
    
    def _update_subscriber_state(self, msisdn: str, record: List, action: int):
        if msisdn not in self.subscriber_state:
            self.subscriber_state[msisdn] = {
                "record_count": 0,
                "spend_history": [],
                "duration_history": [],
                "actions": [],
                "fraud_detected": False,
                "subscription_date": None,
                "sim_change_date": None
            }
        
        state = self.subscriber_state[msisdn]
        state["record_count"] += 1
        
        if len(record) > 21 and record[21]:
            try:
                spend = float(record[21])
                state["spend_history"].append(spend)
            except:
                pass
        
        if len(record) > 17 and record[17]:
            try:
                duration = float(record[17])
                state["duration_history"].append(duration)
            except:
                pass
        
        if len(record) > 22:
            state["subscription_date"] = record[22]
        if len(record) > 23:
            state["sim_change_date"] = record[23]
        
        state["actions"].append(action)
        state["fraud_detected"] = state["fraud_detected"] or (action == 1)
    
    def _extract_features(self, record: List, msisdn: str) -> Dict:
        """Return observable model features only. Ground-truth labels are intentionally excluded."""
        def parse_dt(value: str) -> Optional[datetime.datetime]:
            try:
                return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError):
                return None

        event_time = parse_dt(record[15]) if len(record) > 15 else None
        subscription_time = parse_dt(record[22]) if len(record) > 22 else None
        sim_change_time = parse_dt(record[23]) if len(record) > 23 else None
        CALLED_MSISDN = record[18] if len(record) > 18 else ""

        days_since_subscription = (event_time - subscription_time).days if event_time and subscription_time else -1
        days_since_sim_change = (event_time - sim_change_time).days if event_time and sim_change_time else -1

        features = {
            'USAGE_TYPE': record[0] if len(record) > 0 else "",
            'ACTIVITY_TYPE': record[1] if len(record) > 1 else "",
            'duration': float(record[17]) if len(record) > 17 and record[17] else 0.0,
            'data_volume': float(record[20]) if len(record) > 20 and record[20] else 0.0,
            'charge': float(record[21]) if len(record) > 21 and record[21] else 0.0,
            'tower_environment': record[14] if len(record) > 14 else "",
            'hour': event_time.hour if event_time else -1,
            'day_of_week': event_time.weekday() if event_time else -1,
            'is_weekend': int(event_time.weekday() >= 5) if event_time else 0,
            'days_since_subscription': days_since_subscription,
            'days_since_sim_change': days_since_sim_change,
            'recent_sim_change_7d': int(0 <= days_since_sim_change <= 7),
            'is_international_destination': int(CALLED_MSISDN.startswith('+') and not CALLED_MSISDN.startswith('+27')),
            'is_premium_destination': int(CALLED_MSISDN.startswith('+900') or CALLED_MSISDN in {'44555','44666','44777','44888','44999'}),
            'is_roaming': int((record[14] if len(record) > 14 else '') == 'ROAMING'),
        }

        if msisdn in self.subscriber_state:
            state = self.subscriber_state[msisdn]
            features['record_count'] = state["record_count"]
            features['avg_spend'] = float(np.mean(state["spend_history"])) if state["spend_history"] else 0.0
            features['avg_duration'] = float(np.mean(state["duration_history"])) if state["duration_history"] else 0.0
        else:
            features['record_count'] = 0
            features['avg_spend'] = 0.0
            features['avg_duration'] = 0.0

        return features

    def _get_initial_state(self):
        return {'step': 0, 'total_records': len(self.records)}
    
    def get_metrics(self) -> Dict:
        if self.total_predictions == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        
        tp = self.confusion_matrix["tp"]
        fp = self.confusion_matrix["fp"]
        tn = self.confusion_matrix["tn"]
        fn = self.confusion_matrix["fn"]
        
        accuracy = (tp + tn) / self.total_predictions if self.total_predictions > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_predictions": self.total_predictions,
            "cumulative_reward": self.cumulative_reward,
            "confusion_matrix": self.confusion_matrix
        }

# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    """Main execution function"""
    
    generator = EnhancedReinforcementLearningCDRGenerator(
        num_records=Config.NUM_RECORDS,
        fraud_ratio=Config.FRAUD_RATIO,
        config=Config()
    )
    
    records = generator.generate(Config.OUTPUT_FILE)
        
    env = EnhancedRLEnvironment(generator)
    env.load_records(Config.OUTPUT_FILE)
    
    state = env.reset()
    
    test_steps = min(5000, len(env.records))
    
    for step in range(test_steps):
        current_record = env.records[env.current_step] if env.current_step < len(env.records) else None
        if current_record:
            # Leakage-free baseline: use only observable CDR attributes.
            duration = float(current_record[17]) if current_record[17] else 0.0
            data_volume = float(current_record[20]) if current_record[20] else 0.0
            charge = float(current_record[21]) if current_record[21] else 0.0
            destination = current_record[18] if len(current_record) > 18 else ""
            environment = current_record[14] if len(current_record) > 14 else ""
            risk_points = 0
            risk_points += int(duration > 1200)
            risk_points += int(data_volume > 500)
            risk_points += int(charge > 20)
            risk_points += int(destination.startswith("+900") or destination in {"44555", "44666", "44777", "44888", "44999"})
            risk_points += int(environment == "ROAMING")
            action = 1 if risk_points >= 2 else 0
        else:
            action = 0

        features, reward, done, info = env.step(action)
        if done:
            break
    
    metrics = env.get_metrics()
    
    if generator.device_tracker.sim_swap_events:
        for i, event in enumerate(generator.device_tracker.sim_swap_events[:20]):
            print(f"\nSIM Swap Event {i+1}:")
            print(f"  MSISDN: {event['msisdn']}")
            print(f"  Swap Number: {event['swap_number']}")
            print(f"  Swap Time: {event['swap_time']}")
            print(f"  Subscription Date: {event['subscription_date']}")
            print(f"  Old Device: {event['old_device_name']} ({event['old_device_brand']})")
            print(f"  New Device: {event['new_device_name']} ({event['new_device_brand']})")
            print(f"  Old IMEI: {event['old_imei'][:8]}...{event['old_imei'][-4:]}")
            print(f"  New IMEI: {event['new_imei'][:8]}...{event['new_imei'][-4:]}")
            print(f"  Old IMSI: {event['old_imsi'][:6]}...{event['old_imsi'][-4:]}")
            print(f"  New IMSI: {event['new_imsi'][:6]}...{event['new_imsi'][-4:]}")
    
    print("\n" + "=" * 70)
    print("Generation complete! Dataset ready for Q-TeleGuard training.")
    print("=" * 70)

if __name__ == "__main__":
    main()