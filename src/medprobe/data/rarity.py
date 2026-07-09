"""Common vs rare disease tagging via keyword matching."""
from __future__ import annotations

COMMON_DISEASES = {
    "hypertension", "high blood pressure", "htn",
    "diabetes", "diabetic", "dm", "type 1 diabetes", "type 2 diabetes", "t1d", "t2d",
    "asthma", "pneumonia", "heart failure", "chf",
    "copd", "chronic obstructive",
    "depression", "anxiety", "panic disorder",
    "obesity", "overweight",
    "stroke", "cva", "cerebrovascular",
    "anemia", "iron deficiency",
    "hypothyroidism", "hyperthyroidism", "thyroid",
    "urinary tract infection", "uti",
    "osteoarthritis", "arthritis",
    "migraine", "headache",
    "gerd", "reflux", "acid reflux",
    "atrial fibrillation", "afib", "a. fib", "a-fib",
    "chronic kidney", "ckd",
    "hepatitis", "tuberculosis", "tb",
    "influenza", "flu", "covid", "coronavirus",
    "bronchitis", "epilepsy", "seizure",
    "gout", "psoriasis", "eczema",
    "rheumatoid arthritis",
    "osteoporosis",
    "coronary artery", "cad", "myocardial infarction", "mi", "heart attack",
    "vitamin d deficiency",
    "allergic rhinitis", "hay fever",
    "gallstones", "cholelithiasis",
    "peptic ulcer",
    "pancreatitis", "appendicitis", "diverticulitis",
    "cirrhosis",
    "deep vein thrombosis", "dvt", "pulmonary embolism", "pe",
    "sepsis",
    "breast cancer", "lung cancer", "colon cancer", "prostate cancer",
}

RARE_DISEASES = {
    "addison", "cushing", "pheochromocytoma",
    "wegener", "granulomatosis with polyangiitis", "gpa",
    "goodpasture",
    "myasthenia gravis",
    "guillain-barré", "guillain barre", "gbs",
    "sarcoidosis",
    "amyloidosis",
    "porphyria",
    "wilson disease", "wilson's",
    "hemochromatosis",
    "marfan",
    "ehlers-danlos", "ehlers danlos",
    "fabry", "gaucher",
    "huntington",
    "acromegaly",
    "mastocytosis",
    "cryoglobulinemia",
    "behçet", "behcet",
    "takayasu",
    "polyarteritis nodosa", "pan",
    "dermatomyositis", "polymyositis",
    "scleroderma", "systemic sclerosis",
    "primary biliary", "pbc",
    "budd-chiari", "budd chiari",
    "moyamoya",
    "whipple",
    "churg-strauss", "churg strauss", "eosinophilic granulomatosis",
    "kawasaki",
    "tay-sachs", "tay sachs",
    "pompe",
    "niemann-pick", "niemann pick",
}


def classify_rarity(text: str) -> str:
    """Return 'rare' if any rare keyword matches, 'common' if any common keyword matches,
    else 'unknown'. Rare wins over common if both match (rare diseases are often mentioned
    alongside common ones in distractors)."""
    t = text.lower()
    for kw in RARE_DISEASES:
        if kw in t:
            return "rare"
    for kw in COMMON_DISEASES:
        if kw in t:
            return "common"
    return "unknown"
