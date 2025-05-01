#-----------Data Loading-----------

path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

data = []

for dir_ in os.listdir(BASE_PATH):
    dir_path = os.path.join(BASE_PATH, dir_)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            data.append([dir_, img_path])

df = pd.DataFrame(data, columns=["dir_name", "image_path"])
df_imgs = df[~df["image_path"].str.contains("mask")]
df_masks = df[df["image_path"].str.contains("mask")]

imgs = sorted(df_imgs["image_path"].values, key= lambda x: x[BASE_LEN: -END_LEN])
masks = sorted(df_masks["image_path"].values, key=lambda x: x[BASE_LEN: -END_MASK_LEN])

gene_df = pd.read_csv('/imputed_genomic_data.csv')
gene_df = gene_df.iloc[:, :8]

df_selected_columns = gene_df.iloc[:, 1:8]
combined_array = df_selected_columns.values.tolist()
gene_df['gene_array'] = combined_array

gene_df = gene_df[['Patient','gene_array']]

dir_names = df_imgs['dir_name']

data = []
for image, mask, dir_name in zip(imgs, masks, dir_names):
    data.append({
        'patient': dir_name,  # dir_name을 patient로 사용
        'image_path': image,
        'mask_path': mask
    })
df_patient_images_masks = pd.DataFrame(data)
dff = df_patient_images_masks
dff['patient_id'] = dff['patient'].str[:12]

#print(dff.shape) -> (3929, 4)

def pos_neg_diagnosis(mask_path):
    val = np.max(cv2.imread(mask_path))
    if val > 0: return 1
    else: return 0

gene_dff = pd.merge(dff, gene_df, left_on='patient_id', right_on='Patient', how='left')
gene_dff = gene_dff[['patient','image_path','mask_path','gene_array']]      
gene_dff["diagnosis"] = gene_dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------Data Preprocessing-----------

class BrainMRIDataset:
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)
        gene_info = self.df.iloc[idx, 3]

        augmented = self.transforms(image=image,
                                   mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]
        if len(mask.shape) == 2:  # H, W
          mask = mask.unsqueeze(0)  # 1, H, W

        return image, mask, gene_info

def custom_collate_fn(batch):
    images, masks, gene_infos = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks, list(gene_infos)
  
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

PATCH_SIZE = 128

transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(p=1.0),
    ToTensorV2(),

])

unique_patients = gene_dff['patient'].unique()
train_patients, val_test_patients = train_test_split(unique_patients, stratify=gene_dff.drop_duplicates('patient')['diagnosis'], test_size=0.2, random_state=SEED)
val_patients, test_patients = train_test_split(val_test_patients, stratify=gene_dff[gene_dff['patient'].isin(val_test_patients)].drop_duplicates('patient')['diagnosis'], test_size=0.5, random_state=SEED)

train_df = gene_dff[gene_dff['patient'].isin(train_patients)].reset_index(drop=True)
val_df = gene_dff[gene_dff['patient'].isin(val_patients)].reset_index(drop=True)
test_df = gene_dff[gene_dff['patient'].isin(test_patients)].reset_index(drop=True)

#print(f"Train: {train_df.shape} Val: {val_df.shape} Test: {test_df.shape}")
# -> Train: (3092, 5) Val: (431, 5) Test: (406, 5)

g = torch.Generator()
g.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_dataset = BrainMRIDataset(train_df, transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=26, num_workers=2, shuffle=True, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)

val_dataset = BrainMRIDataset(val_df, transforms=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=26, num_workers=2, shuffle=True, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)

test_dataset = BrainMRIDataset(test_df, transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=26, num_workers=2, shuffle=False, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)
