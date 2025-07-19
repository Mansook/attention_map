def get_class_name_by_index(config, index):
        """
        인덱스를 클래스 이름으로 변환
        """
        if config is None or 'class_map' not in config:
            return f"클래스 {index}"
        
        class_map = {}
        for item in config['class_map']:
            for key, value in item.items():
                class_map[int(key)] = value
        
        return class_map.get(index, f"클래스 {index}")