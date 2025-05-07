import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# Diretório com todos os arquivos (imagens e XMLs)
base_dir = "data/DARTIS_2019_allfiles"
out_img_dir = "yolov5_data/images"
out_lbl_dir = "yolov5_data/labels"

# Cria as pastas de destino
for split in ["train", "val"]:
    os.makedirs(os.path.join(out_img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(out_lbl_dir, split), exist_ok=True)

# Lista todos os arquivos XML
xml_files = [f for f in os.listdir(base_dir) if f.endswith(".xml")]

# Separa train/val (80/20)
train_xmls, val_xmls = train_test_split(xml_files, test_size=0.2, random_state=42)

splits = {"train": train_xmls, "val": val_xmls}

for split, xml_list in splits.items():
    for xml_file in xml_list:
        xml_path = os.path.join(base_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Em vez de confiar no <filename>, usa o mesmo nome base do XML
        img_name = xml_file.replace(".xml", ".jpg")
        img_path = os.path.join(base_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Imagem não encontrada para {xml_file}, pulando...")
            continue

        # Copia imagem para pasta do YOLOv5
        shutil.copy(img_path, os.path.join(out_img_dir, split, img_name))

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        label_path = os.path.join(out_lbl_dir, split, xml_file.replace(".xml", ".txt"))
        with open(label_path, "w") as f:
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name.lower() != "oil":
                    continue
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                # Conversão para YOLO: (x_center, y_center, width, height) normalizados
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
print("Conversion process finished.")
