"""Board capture for same-input Docker/DPU alignment, outside timing benchmarks."""
from __future__ import print_function
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from run_hybrid_board import HybridRunner, load_image
from model_libraries import ensure_library_search_path

HERE = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'runtime_config.json')
    parser.add_argument('--image-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-per-class',type=int,default=2)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    if args.diagnostic_per_class < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid diagnostic-per-class/limit')
    ensure_library_search_path(args.config)
    files = sorted(p for p in args.image_dir.resolve().rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError('No images')
    directory = args.image_dir.resolve()
    labels = []
    for path in files:
        parts = path.relative_to(directory).parts
        if len(parts)<2 or parts[0] not in ['c'+str(i) for i in range(10)]:
            raise ValueError('Need c0...c9 labels: '+str(path))
        labels.append(int(parts[0][1:]))
    if args.output_dir.exists():
        raise ValueError('Capture directory exists; choose a new directory')
    args.output_dir.mkdir(parents=True)
    info = {'status':'failed','samples':[],'error':None,'scope':'Actual board FP32 tensors before/after DPU tasks; not a timing run'}
    per_class = [0]*10
    runner = None
    correct = 0
    try:
        with HybridRunner(args.config,mv2_backend='dpu',fast_tail=True,input_range_policy='error') as runner:
            config = runner.config
            if config.get('mv2_transform') != 'scheme_a_frozen_cle_control':
                raise ValueError('Use the supplied Scheme A deploy_control package')
            info['model_so_hashes'] = {'front':config['front']['so_sha256']}
            info['model_so_hashes'].update({s['name']:s['so_sha256'] for s in config['stages'] if s['backend']=='dpu'})
            for i,(path,label) in enumerate(zip(files,labels)):
                image = load_image(path)
                front = runner.front.predict(image)
                first_input = runner.stages[0][1].predict(front)
                first_output = runner.stages[1][1].predict(first_input)
                second_input = runner.stages[2][1].predict(first_output)
                second_output = runner.stages[3][1].predict(second_input)
                logits = runner.stages[4][1].predict(second_output)
                values = {'mv2_4_input':first_input,'mv2_5_input':second_input,'logits':logits}
                diagnostic = per_class[label] < args.diagnostic_per_class
                if diagnostic:
                    values.update(image=image,front_output=front,mv2_4_output=first_output,mv2_5_output=second_output)
                per_class[label] += 1
                filename = 'capture_{:05d}.npz'.format(i)
                archive = args.output_dir/filename
                np.savez_compressed(str(archive),**values)
                prediction = int(logits.argmax())
                correct += int(prediction==label)
                info['samples'].append({'image':path.relative_to(directory).as_posix(),'label':label,
                    'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'file':filename,
                    'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'diagnostic':diagnostic,
                    'board_class':prediction,'board_logits':logits.reshape(-1).tolist()})
                if (i+1)%25==0:
                    print('Captured {}/{}'.format(i+1,len(files)),flush=True)
            info.update(status='completed',correct_images=correct,accuracy=correct/float(len(files)))
    except BaseException as error:
        info['error'] = '{}: {}'.format(type(error).__name__,error)
        raise
    finally:
        if runner is not None:
            runner.save_input_report(args.output_dir)
        (args.output_dir/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='samples'},indent=2))


if __name__ == '__main__':
    main()
