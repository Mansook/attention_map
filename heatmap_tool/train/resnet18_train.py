import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub

# 상위 디렉토리 추가 (heatmap_tool 폴더)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 이제 상위 디렉토리의 모듈들을 import할 수 있음
from dataset.datasetLoader import get_dataloaders
from models.resnet_classifier import ResNetClassifier
from utils.save_checkpoint_config import save_checkpoint_config

def load_dataset_config(dataset_name):
    """configs/dataset/{dataset_name}.yaml 파일에서 설정 로드"""
    # heatmap_tool 폴더 기준으로 경로 설정
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"configs/dataset/{dataset_name.lower()}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config 파일이 비어있거나 잘못된 형식입니다: {config_path}")
    
    # 필수 필드 검증
    required_fields = ['name', 'root', 'num_classes', 'batch_size']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Config 파일에 필수 필드가 누락되었습니다: {missing_fields}")
    
    return config


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f'[Train] Epoch {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = test_loss / total
    acc = 100. * correct / total
    print(f'[Test ] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description='ResNet18 모델 학습 및 체크포인트 저장')
    parser.add_argument('-d','--dataset', type=str, required=True,
                       help='데이터셋 이름 (STL10, CIFAR10, CIFAR100)')
    parser.add_argument('-b','--batch-size', type=int, default=None,
                       help='배치 크기 (config에서 기본값 사용)')
    parser.add_argument('-e','--epochs', type=int, default=10,
                       help='학습 에포크 수')
    parser.add_argument('-l','--lr', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('-o','--optimizer', type=str, default='Adam',
                       help='옵티마이저 (Adam, SGD)')
    
    args = parser.parse_args()
    
    # 1. config 로드
    config = load_dataset_config(args.dataset)
    print(f"📁 Config 파일 로딩: {args.dataset}")
    print("✅ Config 로드 완료")
    print(f"   - 데이터 경로: {config['root']}")
    print(f"   - 클래스 수: {config['num_classes']}")
    print(f"   - 배치 크기: {config['batch_size']}")

    # 2. kagglehub_id가 있으면 다운로드
    kagglehub_id = config.get('kagglehub_id', None)
    if kagglehub_id:
        # 데이터셋이 없으면 다운로드
        if not os.path.exists(config['root']):
            print(f"데이터셋이 없으므로 kagglehub에서 다운로드합니다... ({kagglehub_id})")
            path = kagglehub.dataset_download(kagglehub_id)
            print("Path to dataset files:", path)
            # config['root']를 다운로드 받은 경로로 덮어쓰기
            config['root'] = path

    # 3. 이후 기존대로 데이터로더 생성
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # ResNet18 모델 설정 가져오기
    model_name = "ResNet18"
    if 'models' not in config or model_name not in config['models']:
        print(f"❌ ResNet18 모델 설정을 config에서 찾을 수 없습니다.")
        return
    
    # 모델 구조를 config에서 가져오기
    model_structure = config['models'][model_name]
    
    # 배치 크기 설정 (명령행 인수가 있으면 우선, 없으면 config 사용)
    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    
    # 체크포인트 이름 자동 생성
    checkpoint_name = f"{args.dataset.lower()}.pt"
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 사용 디바이스: {device}")
    
    # 모델 생성
    print(f" 모델 생성: {model_name}")
    model = ResNetClassifier(num_classes=config['num_classes'], dataset_config=model_structure).to(device)
    
    print(f"✅ 모델 생성 완료 (클래스 수: {config['num_classes']})")
    print(f"model_structure: {model_structure}")
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"학습 시작: {args.epochs} 에포크")
    print(f"옵티마이저: {args.optimizer}, 학습률: {args.lr}")
    
    # 학습 기록
    best_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        print(f'----- Epoch {epoch}/{args.epochs} -----')
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)

        # validation 체크
        if val_loader is not None:
            val_loss, val_acc = test(model, device, val_loader, criterion, epoch)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            print(f'[Valid] Epoch {epoch} | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
            # best_acc는 validation 기준으로 저장
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"새로운 최고 성능! (Validation 기준) 정확도: {best_acc:.2f}%")
        else:
            training_history['val_loss'].append(None)
            training_history['val_acc'].append(None)

        # test 체크 (있으면)
        if test_loader is not None:
            test_loss, test_acc = test(model, device, test_loader, criterion, epoch)
            training_history['test_loss'].append(test_loss)
            training_history['test_acc'].append(test_acc)
        else:
            training_history['test_loss'].append(None)
            training_history['test_acc'].append(None)
    
    # 체크포인트 저장 (heatmap_tool 폴더 기준)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'checkpoints/resnet18/{checkpoint_name}')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f'💾 모델이 {checkpoint_path}로 저장되었습니다.')
    
    # 체크포인트 config 저장
    print("📝 체크포인트 config 저장 중...")
    
    try:
        # 모델 구조와 데이터셋 설정 가져오기
        # dataset yaml(config)에서 model_structure를 직접 가져오도록 변경
        # dataset yaml(config)에서 모델 구조를 직접 가져오도록 변경
        # 예: config['model']['ResNet18']에서 구조를 가져옴
        model_structure = config['models']['ResNet18']
        dataset_config = get_dataloaders(config)
        
        # 학습 정보 구성
        training_info = {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'optimizer': args.optimizer,
            'batch_size': batch_size,
            'best_accuracy': best_acc,
            'final_train_acc': training_history['train_acc'][-1],
            'final_test_acc': training_history['test_acc'][-1],
            'training_history': training_history
        }
        
        # config 저장 (heatmap_tool 폴더 기준)
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/checkpoints/resnet18")
        config_path = save_checkpoint_config(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            dataset_name=args.dataset,
            num_classes=config['num_classes'],
            model_structure=model_structure,
            dataset_config=config,
            training_info=training_info,
            config_dir=config_dir
        )
        
        print(f"✅ Config가 {config_path}에 저장되었습니다.")
    except Exception as e:
        print(f"⚠️  Config 저장 중 오류: {e}")
    
    print(f"\n🎉 === 학습 완료 ===")
    print(f"체크포인트: {checkpoint_path}")
    print(f"최고 성능: {best_acc:.2f}%")
    print(f"XAI 실행 예시:")
    print(f"  python use_xai_explainer_checkpoint.py --checkpoint {checkpoint_name} -x cam")
    print(f"  python use_xai_explainer_checkpoint.py --checkpoint {checkpoint_name} -x grad_cam")

if __name__ == "__main__":
    main() 