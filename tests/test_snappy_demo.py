#!/usr/bin/env python3
"""
Snappy Compression Demo for Vector Database
Demonstrates storage efficiency with multiple documents
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import requests

def create_large_sample_documents():
    """Create large sample documents for testing compression"""
    sample_docs = {
        "medical_encyclopedia": {
            "diabetes_comprehensive.txt": """
Diabetes Management Guide - Comprehensive Medical Reference

Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels due to either insufficient insulin production, ineffective insulin action, or both. This condition affects millions of people worldwide and requires careful management to prevent complications.

Types of Diabetes:
1. Type 1 Diabetes: Autoimmune condition where the pancreas produces little or no insulin
2. Type 2 Diabetes: Most common form, where the body becomes resistant to insulin
3. Gestational Diabetes: Develops during pregnancy and usually resolves after childbirth
4. Prediabetes: Blood sugar levels higher than normal but not high enough for diabetes diagnosis

Pathophysiology:
The pancreas contains beta cells that produce insulin, a hormone essential for glucose metabolism. In diabetes, either these cells are destroyed (Type 1) or the body becomes resistant to insulin's effects (Type 2). This leads to elevated blood glucose levels, which can cause damage to various organs over time.

Clinical Manifestations:
- Polyuria (frequent urination)
- Polydipsia (excessive thirst)
- Polyphagia (increased hunger)
- Unexplained weight loss
- Fatigue and weakness
- Blurred vision
- Slow-healing wounds
- Recurrent infections

Diagnostic Criteria:
- Fasting plasma glucose ≥ 126 mg/dL
- 2-hour plasma glucose ≥ 200 mg/dL during OGTT
- HbA1c ≥ 6.5%
- Random plasma glucose ≥ 200 mg/dL with symptoms

Treatment Approaches:
1. Lifestyle Modifications:
   - Regular physical activity (150 minutes/week)
   - Healthy diet with carbohydrate counting
   - Weight management
   - Smoking cessation
   - Stress management

2. Pharmacological Therapy:
   - Metformin (first-line for Type 2)
   - Sulfonylureas
   - DPP-4 inhibitors
   - GLP-1 receptor agonists
   - SGLT2 inhibitors
   - Insulin therapy (for Type 1 and advanced Type 2)

3. Monitoring:
   - Self-monitoring of blood glucose
   - HbA1c testing every 3-6 months
   - Regular foot examinations
   - Annual comprehensive eye examination
   - Blood pressure monitoring
   - Lipid profile assessment

Complications:
1. Microvascular Complications:
   - Diabetic retinopathy
   - Diabetic nephropathy
   - Diabetic neuropathy

2. Macrovascular Complications:
   - Cardiovascular disease
   - Cerebrovascular disease
   - Peripheral arterial disease

3. Acute Complications:
   - Diabetic ketoacidosis (DKA)
   - Hyperosmolar hyperglycemic state (HHS)
   - Hypoglycemia

Prevention Strategies:
- Regular screening for high-risk individuals
- Early intervention for prediabetes
- Public health education
- Community-based prevention programs
- Healthcare provider training

Emergency Management:
- Recognition of DKA symptoms
- Immediate medical attention for severe hyperglycemia
- Hypoglycemia treatment protocols
- Emergency contact information

Patient Education:
- Self-management skills
- Medication adherence
- Dietary counseling
- Exercise recommendations
- Foot care instructions
- Blood glucose monitoring techniques

Quality of Life Considerations:
- Psychological support
- Social support networks
- Vocational rehabilitation
- Family education
- Community resources

Research and Innovation:
- Continuous glucose monitoring
- Artificial pancreas systems
- New medication classes
- Stem cell therapy research
- Gene therapy approaches

This comprehensive guide provides the foundation for understanding diabetes management and should be used in conjunction with professional medical advice.
            """,
            "cardiology_advanced.txt": """
Advanced Cardiology - Comprehensive Heart Health Reference

Cardiovascular disease remains the leading cause of death globally, making understanding of cardiac physiology, pathology, and treatment essential for healthcare professionals and patients alike.

Cardiac Anatomy and Physiology:
The heart is a muscular organ located in the thoracic cavity, functioning as a dual pump system. The right side receives deoxygenated blood from the body and pumps it to the lungs, while the left side receives oxygenated blood from the lungs and pumps it to the body.

Chamber Structure:
- Right Atrium: Receives blood from superior and inferior vena cava
- Right Ventricle: Pumps blood to pulmonary circulation
- Left Atrium: Receives oxygenated blood from pulmonary veins
- Left Ventricle: Pumps blood to systemic circulation

Valve System:
- Tricuspid Valve: Between right atrium and ventricle
- Pulmonary Valve: Between right ventricle and pulmonary artery
- Mitral Valve: Between left atrium and ventricle
- Aortic Valve: Between left ventricle and aorta

Electrical Conduction System:
- Sinoatrial (SA) Node: Natural pacemaker
- Atrioventricular (AV) Node: Delays conduction
- Bundle of His: Conducts to ventricles
- Purkinje Fibers: Distributes electrical signals

Common Cardiovascular Conditions:

1. Coronary Artery Disease (CAD):
   - Atherosclerosis of coronary arteries
   - Angina pectoris symptoms
   - Myocardial infarction risk
   - Diagnostic testing: ECG, stress test, angiography
   - Treatment: Medications, angioplasty, bypass surgery

2. Heart Failure:
   - Systolic vs diastolic dysfunction
   - Left vs right heart failure
   - NYHA classification system
   - Treatment: ACE inhibitors, beta-blockers, diuretics
   - Advanced therapies: LVAD, heart transplantation

3. Arrhythmias:
   - Atrial fibrillation management
   - Ventricular tachycardia treatment
   - Bradycardia interventions
   - Antiarrhythmic medications
   - Catheter ablation procedures

4. Valvular Heart Disease:
   - Stenosis vs regurgitation
   - Rheumatic fever complications
   - Degenerative valve disease
   - Surgical vs transcatheter interventions

5. Hypertension:
   - Primary vs secondary causes
   - Target organ damage assessment
   - Lifestyle modifications
   - Pharmacological treatment
   - Resistant hypertension management

Diagnostic Modalities:
- Electrocardiography (ECG)
- Echocardiography
- Cardiac catheterization
- Nuclear imaging
- Cardiac MRI
- CT angiography
- Stress testing

Treatment Approaches:
1. Medical Therapy:
   - Antiplatelet agents
   - Anticoagulation
   - Beta-blockers
   - ACE inhibitors
   - Statins
   - Diuretics

2. Interventional Procedures:
   - Percutaneous coronary intervention
   - Stent placement
   - Balloon angioplasty
   - Transcatheter valve replacement

3. Surgical Interventions:
   - Coronary artery bypass grafting
   - Valve replacement/repair
   - Heart transplantation
   - Ventricular assist devices

Prevention Strategies:
- Primary prevention programs
- Risk factor modification
- Regular screening
- Public health initiatives
- Patient education

Emergency Management:
- Acute coronary syndrome protocols
- Cardiac arrest algorithms
- Heart failure decompensation
- Hypertensive emergencies
- Arrhythmia management

Quality Metrics:
- Door-to-balloon time
- Readmission rates
- Mortality statistics
- Patient satisfaction scores
- Functional outcomes

Research Frontiers:
- Stem cell therapy
- Gene therapy approaches
- Artificial heart development
- Minimally invasive techniques
- Precision medicine applications

This comprehensive cardiology reference provides essential knowledge for understanding and managing cardiovascular health.
            """,
            "neurology_basics.txt": """
Neurology Fundamentals - Comprehensive Nervous System Reference

The nervous system is the body's electrical communication network, coordinating all bodily functions and enabling complex behaviors. Understanding neurology is essential for diagnosing and treating a wide range of conditions.

Nervous System Anatomy:
The nervous system consists of the central nervous system (CNS) and peripheral nervous system (PNS). The CNS includes the brain and spinal cord, while the PNS includes all nerves outside the CNS.

Brain Structure:
- Cerebrum: Largest part, responsible for higher functions
- Cerebellum: Coordinates movement and balance
- Brainstem: Controls vital functions
- Diencephalon: Includes thalamus and hypothalamus

Spinal Cord:
- Cervical region (C1-C8)
- Thoracic region (T1-T12)
- Lumbar region (L1-L5)
- Sacral region (S1-S5)
- Coccygeal region

Common Neurological Conditions:

1. Stroke:
   - Ischemic vs hemorrhagic
   - Risk factors and prevention
   - Acute management protocols
   - Rehabilitation strategies
   - Secondary prevention

2. Epilepsy:
   - Seizure classification
   - Diagnostic testing
   - Antiepileptic medications
   - Surgical options
   - Lifestyle modifications

3. Multiple Sclerosis:
   - Autoimmune demyelination
   - Relapsing-remitting course
   - Disease-modifying therapies
   - Symptom management
   - Quality of life considerations

4. Parkinson's Disease:
   - Dopamine deficiency
   - Motor and non-motor symptoms
   - Levodopa therapy
   - Deep brain stimulation
   - Supportive care

5. Alzheimer's Disease:
   - Amyloid plaques and tau tangles
   - Memory and cognitive decline
   - Cholinesterase inhibitors
   - Behavioral management
   - Caregiver support

Diagnostic Tools:
- Neurological examination
- Imaging studies (CT, MRI)
- Electroencephalography (EEG)
- Electromyography (EMG)
- Nerve conduction studies
- Lumbar puncture

Treatment Modalities:
1. Pharmacological:
   - Antiepileptic drugs
   - Dopaminergic agents
   - Cholinesterase inhibitors
   - Immunomodulatory therapies
   - Pain management

2. Surgical:
   - Deep brain stimulation
   - Epilepsy surgery
   - Tumor resection
   - Aneurysm clipping
   - Spinal procedures

3. Rehabilitation:
   - Physical therapy
   - Occupational therapy
   - Speech therapy
   - Cognitive rehabilitation
   - Assistive devices

Emergency Neurology:
- Status epilepticus
- Acute stroke protocols
- Increased intracranial pressure
- Spinal cord compression
- Neuromuscular emergencies

Prevention Strategies:
- Stroke prevention
- Head injury prevention
- Lifestyle modifications
- Regular screening
- Public health education

Research Advances:
- Neuroimaging techniques
- Biomarker development
- Gene therapy approaches
- Stem cell research
- Artificial intelligence applications

This comprehensive neurology reference provides essential knowledge for understanding nervous system function and disorders.
            """
        }
    }
    
    # Create directories and files
    for domain, documents in sample_docs.items():
        domain_dir = Path(f"large_test_documents/{domain}")
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in documents.items():
            file_path = domain_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            print(f"✅ Created: {file_path}")

def test_snappy_compression_demo():
    """Test Snappy compression with large documents"""
    print("🧪 Snappy Compression Demo with Large Documents")
    print("=" * 60)
    
    # Create large sample documents
    print("\n📄 Creating large sample documents...")
    create_large_sample_documents()
    
    # Test domain
    domain = "medical_encyclopedia"
    
    # Create domain
    try:
        response = requests.post(
            "http://localhost:8000/api/upload/domain",
            data={"domain": domain}
        )
        
        if response.status_code == 200:
            print(f"✅ Domain {domain} created")
        else:
            print(f"❌ Failed to create domain: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error creating domain: {e}")
        return
    
    # Upload documents one by one to see compression
    test_files = [
        "large_test_documents/medical_encyclopedia/diabetes_comprehensive.txt",
        "large_test_documents/medical_encyclopedia/cardiology_advanced.txt",
        "large_test_documents/medical_encyclopedia/neurology_basics.txt"
    ]
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n📤 Uploading document {i}/{len(test_files)}: {Path(file_path).name}")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'domain': domain,
                    'metadata': json.dumps({
                        "snappy_compression_test": True,
                        "document_number": i,
                        "file_size": Path(file_path).stat().st_size
                    })
                }
                
                response = requests.post(
                    "http://localhost:8000/api/upload/doc",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Document uploaded successfully!")
                print(f"   📄 File: {Path(file_path).name}")
                print(f"   📊 Documents added: {result.get('documents_added')}")
                
                # Check vectorstore size after each upload
                vectorstore_path = Path("vectorstore") / domain
                if vectorstore_path.exists():
                    total_size = sum(f.stat().st_size for f in vectorstore_path.rglob('*') if f.is_file())
                    print(f"   💾 Vectorstore size: {total_size / (1024 * 1024):.2f} MB")
                
            else:
                print(f"❌ Upload failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error uploading document: {e}")
    
    # Final analysis
    print(f"\n📊 Final Storage Analysis:")
    vectorstore_path = Path("vectorstore") / domain
    if vectorstore_path.exists():
        total_size = sum(f.stat().st_size for f in vectorstore_path.rglob('*') if f.is_file())
        print(f"   📁 Total vectorstore size: {total_size / (1024 * 1024):.2f} MB")
        
        # List all files
        print(f"   📄 Files in vectorstore:")
        for file_path in vectorstore_path.rglob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"      • {file_path.name}: {size_mb:.3f} MB")
                
                # Check for Parquet files
                if file_path.suffix == '.parquet':
                    print(f"         🗜️ Snappy compressed Parquet file")
                elif file_path.suffix == '.sqlite3':
                    print(f"         🗄️ SQLite index file")
                elif file_path.suffix == '.bin':
                    print(f"         📦 Binary data file")
    
    # Get domain stats
    try:
        stats_response = requests.get(f"http://localhost:8000/api/upload/domain/{domain}/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"\n📊 Domain Statistics:")
            print(f"   📄 Document count: {stats.get('stats', {}).get('count', 0)}")
            print(f"   🧠 Embedding model: {stats.get('stats', {}).get('embedding_model', 'unknown')}")
            print(f"   📏 Chunk size: {stats.get('stats', {}).get('chunk_size', 0)}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

if __name__ == "__main__":
    test_snappy_compression_demo() 