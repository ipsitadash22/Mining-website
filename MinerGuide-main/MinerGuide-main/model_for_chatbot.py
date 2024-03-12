import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Prepare your training and validation datasets
question= ["What is the MMRD Act?",
        "What does Section 2 of the MMRD Act declare?",
        "What are the Existing Legislative Provisions regarding safety, health and welfare of mine workers? ",
        "How the Compliance of the Provisions are ensured?",
        "How the Provisions of Health, Safety and Welfare Amenities are enforced?",
        "What is the Role and Function of DGMS?",
        "What are the provisions of law in respect of accidents in mines?",
        "What are the major cases of accidents in mines?",
        "What steps are being taken by the Government to prevent such accidents in mines?",
        "What are the Remedial measures taken to bring down the rate of accidents in Mines?",
        "What are the Legislative Provisions relating to Safety and Health in industries?",
        "What are the important provisions in the Factories Act?",
        "What are the powers of the Inspectors appointed under the Factories Act, 1948?",
        "What is the system of reporting of occupational diseases in the Factories?",
        "What actions are initiated against the management for violation of the provisions of the Act?",
        "What are the major initiatives taken by DGFASLI to improve safety and health of the workers in the manufacturing sector?",
        "How are occupational diseases reported and managed under the Factories Act?",
        "What is the significance of the Factories Act in ensuring workplace safety in industries?",
        "How does the Mines Rescue Rules, 1985, address emergency response in mining accidents?",
        "What is the role of the Coal Mines Welfare Commissioner in ensuring the welfare of coal miners?"]
answer = [
        "The MMRD Act stands for Mines and Minerals (Development and Regulation) Act.",
        "Section 2 of the MMRD Act declares that it is expedient in the public interest for the Union to control the regulation of mines and mineral development.",
        '''Under the Constitution of India, safety, welfare and health of workers employed in mines are the concern of the Central Government (Entry 55- Union List- Article 246).The objective is regulated by the Mines Act, 1952 and the Rules and Regulations framed thereunder which are administered by the 
        Directorate- General of Mines Safety (DGMS), under the Union Ministry of Labour and Employment. A list of the subordinate legislation under the Mines Act administered 
        by DGMS are – 
        • Coal Mines Regulations, 1957. 
        • Metalliferous Mines Regulations, 1961. 
        • Oil Mines Regulations, 1984. 
        • Mines Rules, 1955. 
        • Mines Vocational Training Rules, 1966. 
        • Mines Rescue Rules, 1985. 
        • Mines Creche Rules, 1966.''',
        '''The owner, agent or manager of the mine is required to comply with the 
provisions of health and safety provisions of the Mines Act and the rules 
framed thereunder, as required under Section 18 of the Mines Act, 1952. ''',
'''DGMS is the enforcement agency which ensures compliance of the stated 
provisions through inspections by inspecting officers. The health, safety 
and welfare provisions of Mines Act and Rules are invariably checked 
during the course of general inspection of the mines. The violations 
observed during the course of general inspection of the mines. The 
violations observed during the course of such inspections are being 
followed up by subsequent follow up inspection. In case of nonstxcompliances, the improvement notices, prohibitory orders etc. are also 
being issued till it is complied.''',
'''1. Inspection of mines. 
2. Investigation into – 
 a) accidents 
 b) dangerous occurrences – emergency response 
 c) complaints & other matters 
3. a) Grant of : 
 i) statutory permission, exemptions & relaxations 
 - pre-view of project reports & mining plans 
 ii) approval of mine safety equipment, material & appliances 
 b) Interactions for development of safety equipment, material 
and safe work practices through workshop etc. 
 c) Development of Safety Legislation & Standards 
 d) Safety Information Dissemination.
 4. Conduct of examinations for grant of competency certificates. 
5. Safety promotion initiatives including : 
 (a) Organisation of – 
 Conference on Safety in Mines 
 National Safety Awards 
 Safety Weeks & Campaigns. 
 (b) Promoting – 
 - Safety education and awareness programmes 
 - Workers’ participation in safety management through- 
 Workmen’s inspector 
Safety committee 
Tripartite reviews''',
'''Following provisions are existing in the Mines Act & the Rules & 
Regulations made thereunder on accidents in mines: 
Section 23 of the Mines Act5, 1952: Notice of Accidents 
Notice of accidents by the mine management of DGMS 
Enquiry in to such accident by DGMS 
 Regulation 9: Prescribes nature of accidents and the forms in which notices are to 
be sent to specified persons which include Coal Mines Welfare 
Commissioner in cases of Fatal and Serious accidents. 
 Regulation 199: Places of accidents not to be disturbed unless otherwise permitted 
by Chief Inspector or Inspector. 
 Regulation 199A:Enforcement of Emergency Plan in the mine immediately after 
occurrence of accident. 
 Section 24: Power of Central Govt. to appoint court of inquiry in cases of 
Accidents: 
 Central Government normally appoints court of inquiries in cases of 
major accidents and disasters in mines.''',
'''Coal mines are considered more risky than Metalliferous mines all over the 
world. The incidences of accidents and number of fatality in coal mines 
are higher than non-coal mines. 
 The major causes of accidents in mines are :- 
• Explosions and Fires : Methane & Coal Dust Explosions 
 Spontaneous Heating of Coal 
• Inundation (Sudden inrush of water into the mines: From surface 
 Underground 
• Strata Failure : Roof and Side Fall in Underground Mines 
Pit and Dump Slides & Failure in opencast mines 
• Heavy Earth Moving machinery : Shovel, Dumper, Trucks & Tippers''',
'''(i) All fatal and serious accidents including dangerous occurrences 
especially due to fires, explosives, gases and many other important 
subjects are enquired by DGMS. 
 (ii) After completion of enquiries, legal actions as deem fit including 
prosecution against the persons found responsible for the accidents are 
taken. 
(iii) Accidents are also technically analyzed in details and based on 
findings of such analysis, technical circulars, instructions and guidelines 
are issued on various causes and failures to improve the standards of 
safety in mines and to prevent such recurrences. 
(iv) Accident Prone Mines are also identified on the basis of such 
analysis and focal attentions are given on such mines through 
inspections and follow up action so that their conditions are brought to 
safe levels.''',
'''(i) Strict enforcement of existing statute. 
 (ii) Close monitoring of the working of the mines by Safety Supervisors in the 
mines, Internal Safety Organisation of the mining companies and by the 
Inspecting officers of DGMS. 
 (iii) Taking suitable actions as per the statute for non-compliance such as 
stoppage of work, issue of violation letters, issue of prohibitory 
notices/orders, launching of prosecutions under the court of law etc. 
(iv) Strengthening the mechanism of training & re-training or workers & 
supervisors. 
(v) Inquiry into accidents, analysis for ascertaining the causes and 
circumstances leading to accidents and taken suitable action for preventing 
similar accidents in future. 
(vi) Introducing the concept of Safety Management through risk assessment for 
identification of hazards, assessment of risks in the hazards, evolving 
control measures, implementation of control measures and monitoring the 
effectiveness of the control measures through safety audit. This is a new 
concept and is being introduced gradually in conjunction with existing 
practices of legislative safety management. Workers at all levels are 
involved in the process of decision making on risk management for its 
effective implementation through greater involvement. 
(vii) Improving the awareness of workers at all levels regarding safety issues 
involved in the work process and the safe operating procedures for each 
job.''',
'''The safety, health and welfare or workers employed in factories are 
covered under the Factories Act, 1948 which is a central legislation. The 
Act contains detailed provisions on health, safety welfare, working hours, 
leave, penalties etc. and is applicable to premises wherein 10 or more 
workers are employed without the aid of power. 
 The State Governments are empowered under Section 85 of the Act to 
bring those factories wherein less than 10 workers with the aid lf power or 
20 or more workers are employed without the aid of power under the 
purview of this Act. 
 The provisions of the Factories Act and Rules framed thereunder are 
enforced by the State Governments through the State Factories 
Directorate/Inspectorates.''',
'''The important provisions in the Factories Act, 1948 relates to 
• Appointment of Inspectors, 
• Responsibility of the Occupier and Manufacturer of Articles 
used in factories, (This provisions was incorporated in 1987 
after the Bhopal Tragedy) 
• Health Provisions 
• Safety provisions 
• Welfare Provisions 
• Working Hours. 
• Employment of Young Persons. 
• Annual Leave With Wages. 
• Special Provisions (power to apply the Act to certain premises, 
dangerous operations, notice of accidents and occupational 
diseases, power of enquiry, etc.) 
• Penalties and Procedures. 
• The important provisions relating to Safety and Health of 
workers are given below. 
Health Provisions
Every factory must take the following measures as per the provisions of the 
Act to ensure health of the workers. 
• To keep its premises in a clean state; 
• To dispose of wastes and effluents: 
• To maintain adequate ventilation and reasonable temperature; 
• To prevent accumulation of dust and fume; 
• To avoid over crowding; 
• To provide sufficient lighting, drinking water, latrines and urinals. 
Safety Provisions
 Every factory must take the following measures as per the provisions 
of the Act to ensure safety of the workers? 
• to fence certain machinery; 
• to protect workers repairing machinery in motion; 
• to protect young persons working on dangerous machines; 
• to ensure hoists and lifts and pressure vessels are of sound construction 
and maintained in good working conditions; 
• Floors, stairs and means of access in every factory shall be of sound 
construction and properly maintained to ensure safety of the works. 
• to protect workers from injury to their eyes; 
• to protect workers from dangerous dust, gas, fumes and vapours; 
• to protect workers from fire, explosives or flammable dust or gas, etc.''',
''' An inspector appointed under the Act has power- 
• to enter any place which is used as a factory; 
• to make examination of the premises, plant and machinery. 
• to require the production of any register and any other document relating 
to the factory , and 
• to take statement of any person, for carrying out the purposes of the Act. 
• To initiate legal action for violation or non compliance of the provisions of 
the Act and Rules made thereunder.''',
'''Where any workers in a factory contacts any notifiable disease as specified 
in the Third Schedule the manager of the factory shall send a notice to 
inspector of factories in such a form and in the manner prescribed (Section 
89). ''',
'''The inspectors visit the factories and violations of the provisions of the Act 
and the Rules framed thereunder are brought to the notice of the 
occupier/manager for taking necessary actions particularly when building, 
machineries and equipment are likely to lead conditions detrimental to the 
health and safety of the workers. 
The inspectors also have power to prohibit employment on account of 
serious hazards, initially for a period of three days. 
The occupier is directed to remove the hazard before re-employing the 
workers. 
In case the occupier/manager do not abide by the written order issued by 
the inspector prosecution is initiated for the violation of any of the 
provisions of Act and Rules. (Powers of Inspectors are given in Section 9, 
40-A and Section 87-A).''',
'''The major initiatives undertaken by DGFASLI are: 
• DGFASLI undertook the framing of model factories rules in 
consultation with the Chief Inspector of Factories/Union Territories 
for guidance and adoption by the State Governments to ensure 
uniformity. 
• Organizes annual conference of Chief Inspector of factories. 
• National and consultancy studies are undertaken to assess the status 
of occupational health of the workers in factories and ports to 
formulate appropriate standards/guidelines for inclusion in the 
statutes. 
• Conducts professional and academic training programmes for 
supervisors, safety officers, factory medical officers, specialised 
certificate course for competent supervisions in hazardous process 
industries.''',
'''The Factories Act requires the reporting of occupational diseases. When a worker contracts a notifiable disease,
it must be reported to the inspector of factories, and necessary measures are taken to prevent further occurrences.''',
'''The Factories Act contains provisions related to the safety, health, and welfare of workers in industrial establishments.
It sets standards for workplace safety and ensures that employers provide a safe working environment.''',
'''The Mines Rescue Rules, 1985, lay down procedures for handling emergencies,
including rescue and recovery operations in the event of accidents or disasters in mines,
ensuring the safety of miners.''',
'''The Coal Mines Welfare Commissioner is responsible for implementing welfare schemes for coal miners,
including housing, medical facilities, education, and social amenities, to improve their quality of life.''']

df = pd.DataFrame(data)
df.to_csv('qa_dataset.csv', index=False)

# Split data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

# Step 2: Create a PyTorch DataLoader for your dataset
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['question']
        answer = self.data.iloc[idx]['answer']
        encoding = self.tokenizer(question, answer, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = QADataset(train_df, tokenizer)
valid_dataset = QADataset(valid_df, tokenizer)

batch_size = 8  # Adjust batch size based on available GPU memory
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Step 3: Define and fine-tune your question-answering model
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

# Early stopping parameters
patience = 3
best_valid_loss = float('inf')
counter = 0

# Training loop
num_epochs =5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            valid_loss += loss.item()

    # Calculate average losses
    avg_train_loss = total_loss / len(train_dataloader)
    avg_valid_loss = valid_loss / len(valid_dataloader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}')

    # Early stopping
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        counter = 0
        # Save the best model
        model.save_pretrained('qa_model')
        tokenizer.save_pretrained('qa_model')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print("Training finished.")

# Step 4: Model is saved automatically if validation loss improves during training.
