# gpu_runner.py  
import sys  
import os  
from pathlib import Path  
  
def main():  
    """Script para ejecutar una repetición en una GPU específica"""  
    if len(sys.argv) != 4:  
        print("Uso: python gpu_runner.py <gpu_id> <rep_index> <exp_name>")  
        sys.exit(1)  
      
    gpu_id = int(sys.argv[1])  
    rep_index = int(sys.argv[2])  
    exp_name = sys.argv[3]  
      
    # CRÍTICO: Configurar GPU ANTES de importar TensorFlow  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  
      
    # Configurar rutas (ajustar según tu entorno)  
    BASE_DIR = Path('/kaggle/working/structure')  
      
    # Agregar rutas al sys.path  
    sys.path.append(str(BASE_DIR))  
    sys.path.append(str(BASE_DIR / 'models'))  
      
    # Ahora importar TensorFlow y módulos del framework  
    import tensorflow as tf  
    from utils.experiment.functions import load_config, load_experiment  
    from utils.analysis.analysis import ExperimentAnalyzer  
      
    print(f"🔄 Rep {rep_index+1} iniciando en GPU {gpu_id}")  
    print(f"GPUs disponibles: {len(tf.config.list_physical_devices('GPU'))}")  
      
    try:  
        # Cargar configuración  
        cfg = load_config(exp_name)  
        k = cfg["dataset"].get("k_folds")  
          
        if k is None or k <= 1:  
            # Flujo single-split  
            cfg, NNClass, params, dataset, train_data, val_data, test_idx = \  
                load_experiment(exp_name, repeat_index=rep_index)  
              
            # Verificar si ya existe  
            rep_report = BASE_DIR / cfg['experiment']['output_root'] / cfg['experiment']['output_subdir'] / "reports" / "classification_report.json"  
            if rep_report.exists():  
                print(f"[SKIP] Rep: {rep_index} (GPU {gpu_id}) → ya existe classification_report.json.")  
                return  
              
            # Entrenar modelo  
            model = NNClass(cfg, **params)  
            print(f"🚀 Entrenando Rep {rep_index+1} en GPU {gpu_id}")  
            history = model.fit(train_data, val_data)  
              
            # Análisis  
            analyzer = ExperimentAnalyzer(  
                model=model.model,  
                history=history,  
                test_data=test_idx,  
                cfg=cfg,  
                effects=dataset.get_effects("test"),  
                repeat_index=rep_index,  
                show_plots=False,  
            )  
              
            analyzer.classification_report()  
            analyzer.effect_report()  
            analyzer.confusion_matrix(normalize="true")  
            model.cleanup_old_checkpoints()  
              
            print(f"✅ Rep {rep_index+1} completada en GPU {gpu_id}")  
              
        else:  
            # Flujo K-fold  
            for fold in range(k):  
                print(f"🔄 Rep {rep_index+1} Fold {fold+1} en GPU {gpu_id}")  
                  
                cfg, NNClass, params, dataset, train_data, val_data, test_idx = \  
                    load_experiment(exp_name, repeat_index=rep_index, fold_index=fold)  
                  
                # Verificar si ya existe  
                rep_report = BASE_DIR / cfg['experiment']['output_root'] / cfg['experiment']['output_subdir'] / "reports" / "classification_report.json"  
                if rep_report.exists():  
                    print(f"[SKIP] Rep: {rep_index} Fold: {fold} (GPU {gpu_id}) → ya existe.")  
                    continue  
                  
                # Entrenar modelo  
                model = NNClass(cfg, **params)  
                history = model.fit(train_data, val_data)  
                  
                # Análisis  
                analyzer = ExperimentAnalyzer(  
                    model=model.model,  
                    history=history,  
                    test_data=test_idx,  
                    cfg=cfg,  
                    repeat_index=rep_index,  
                    fold_index=fold,  
                    effects=dataset.get_effects("test"),  
                    show_plots=False,  
                )  
                  
                analyzer.classification_report()  
                analyzer.effect_report()  
                analyzer.confusion_matrix(normalize="true")  
                model.cleanup_old_checkpoints()  
                  
                print(f"✅ Rep {rep_index+1} Fold {fold+1} completada en GPU {gpu_id}")  
                  
    except Exception as e:  
        print(f"❌ Error en Rep {rep_index+1} GPU {gpu_id}: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)  
  
if __name__ == '__main__':  
    main()