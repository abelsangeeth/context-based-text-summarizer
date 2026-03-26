from PIL import Image
import traceback
from input_handlers.image_handler import _detect_objects

try:
    print('Starting detection...')
    img = Image.new('RGB', (640, 480), color='white')
    res = _detect_objects(img, 0.5)
    print(f'Done, got {len(res)} detections.')
except Exception as e:
    print('Error occurred:')
    traceback.print_exc()
