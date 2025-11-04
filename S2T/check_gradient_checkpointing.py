"""
Script để kiểm tra gradient checkpointing implementation
"""
import torch
import torch.nn as nn
from speech2text_model import (
    SeamlessM4Tv2ForSpeechToTextTrain_Pivot,
    SeamlessM4Tv2SpeechEncoder,
    SeamlessM4Tv2Encoder,
    SeamlessM4Tv2Decoder,
    SeamlessM4Tv2ConformerEncoder,
    SeamlessM4Tv2ConformerEncoderLayer,
    SeamlessM4Tv2EncoderLayer,
    SeamlessM4Tv2DecoderLayer,
)
from seamless_m4t_v2_config import SeamlessM4Tv2Config
from utils import GradientCheckpointingLayer

def check_gradient_checkpointing():
    """Kiểm tra gradient checkpointing implementation"""
    
    print("=" * 80)
    print("KIỂM TRA GRADIENT CHECKPOINTING IMPLEMENTATION")
    print("=" * 80)
    
    config = SeamlessM4Tv2Config()
    
    # 1. Kiểm tra các lớp có inherit từ GradientCheckpointingLayer
    print("\n1. Kiểm tra inheritance từ GradientCheckpointingLayer:")
    print("-" * 80)
    
    layer_classes = [
        ("SeamlessM4Tv2ConformerEncoderLayer", SeamlessM4Tv2ConformerEncoderLayer),
        ("SeamlessM4Tv2EncoderLayer", SeamlessM4Tv2EncoderLayer),
        ("SeamlessM4Tv2DecoderLayer", SeamlessM4Tv2DecoderLayer),
    ]
    
    for name, layer_class in layer_classes:
        is_gc_layer = issubclass(layer_class, GradientCheckpointingLayer)
        has_gc_attr = hasattr(layer_class, 'gradient_checkpointing')
        status = "✅" if is_gc_layer and has_gc_attr else "❌"
        print(f"{status} {name}:")
        print(f"   - Inherit từ GradientCheckpointingLayer: {is_gc_layer}")
        print(f"   - Có attribute 'gradient_checkpointing': {has_gc_attr}")
    
    # 2. Kiểm tra các encoder/decoder có method gradient_checkpointing_enable
    print("\n2. Kiểm tra method gradient_checkpointing_enable:")
    print("-" * 80)
    
    # Tạo các instance để kiểm tra
    try:
        speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        text_encoder = SeamlessM4Tv2Encoder(config)
        text_decoder = SeamlessM4Tv2Decoder(config)
        
        encoders = [
            ("SpeechEncoder", speech_encoder),
            ("TextEncoder", text_encoder),
            ("TextDecoder", text_decoder),
        ]
        
        for name, encoder in encoders:
            has_method = hasattr(encoder, 'gradient_checkpointing_enable')
            has_gc_attr = hasattr(encoder, 'gradient_checkpointing')
            is_pretrained = hasattr(encoder, 'supports_gradient_checkpointing')
            
            status = "✅" if has_method else "❌"
            print(f"{status} {name}:")
            print(f"   - Có method 'gradient_checkpointing_enable': {has_method}")
            print(f"   - Có attribute 'gradient_checkpointing': {has_gc_attr}")
            print(f"   - Là PreTrainedModel (supports_gradient_checkpointing): {is_pretrained}")
            
            if has_method:
                try:
                    # Test gọi method
                    encoder.gradient_checkpointing_enable()
                    gc_enabled = encoder.gradient_checkpointing
                    print(f"   - ✅ Method hoạt động, gradient_checkpointing = {gc_enabled}")
                    
                    # Kiểm tra các layers bên trong
                    if hasattr(encoder, 'layers'):
                        if len(encoder.layers) > 0:
                            first_layer = encoder.layers[0]
                            layer_gc = getattr(first_layer, 'gradient_checkpointing', None)
                            print(f"   - Layer gradient_checkpointing: {layer_gc}")
                    
                except Exception as e:
                    print(f"   - ❌ Lỗi khi gọi method: {e}")
        
        # 3. Kiểm tra ConformerEncoder (không phải PreTrainedModel)
        print("\n3. Kiểm tra ConformerEncoder (nn.Module):")
        print("-" * 80)
        
        conformer_encoder = SeamlessM4Tv2ConformerEncoder(config)
        has_method = hasattr(conformer_encoder, 'gradient_checkpointing_enable')
        has_gc_attr = hasattr(conformer_encoder, 'gradient_checkpointing')
        
        status = "⚠️" if not has_method else "✅"
        print(f"{status} ConformerEncoder:")
        print(f"   - Có method 'gradient_checkpointing_enable': {has_method}")
        print(f"   - Có attribute 'gradient_checkpointing': {has_gc_attr}")
        
        if has_gc_attr:
            print(f"   - gradient_checkpointing = {conformer_encoder.gradient_checkpointing}")
        
        # Kiểm tra layers bên trong
        if hasattr(conformer_encoder, 'layers') and len(conformer_encoder.layers) > 0:
            first_layer = conformer_encoder.layers[0]
            layer_gc = getattr(first_layer, 'gradient_checkpointing', None)
            print(f"   - Layer gradient_checkpointing: {layer_gc}")
        
        # 4. Kiểm tra full model
        print("\n4. Kiểm tra Full Model:")
        print("-" * 80)
        
        model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
        
        components = [
            ("model.speech_encoder", model.speech_encoder),
            ("model.text_encoder", model.text_encoder),
            ("model.text_decoder", model.text_decoder),
        ]
        
        for name, component in components:
            has_method = hasattr(component, 'gradient_checkpointing_enable')
            print(f"{'✅' if has_method else '❌'} {name}: có gradient_checkpointing_enable = {has_method}")
        
        # 5. Test enable gradient checkpointing
        print("\n5. Test Enable Gradient Checkpointing:")
        print("-" * 80)
        
        try:
            model.gradient_checkpointing_enable()
            print("✅ Model.gradient_checkpointing_enable() thành công")
            
            # Kiểm tra các components
            for name, component in components:
                gc_enabled = getattr(component, 'gradient_checkpointing', False)
                print(f"   - {name}.gradient_checkpointing = {gc_enabled}")
                
                # Kiểm tra layers
                if hasattr(component, 'layers'):
                    if len(component.layers) > 0:
                        first_layer = component.layers[0]
                        layer_gc = getattr(first_layer, 'gradient_checkpointing', None)
                        print(f"     → Layer gradient_checkpointing = {layer_gc}")
            
            # Kiểm tra ConformerEncoder bên trong SpeechEncoder
            if hasattr(model.speech_encoder, 'encoder'):
                conformer_enc = model.speech_encoder.encoder
                conformer_gc = getattr(conformer_enc, 'gradient_checkpointing', None)
                print(f"   - speech_encoder.encoder.gradient_checkpointing = {conformer_gc}")
                
                # Kiểm tra layers trong ConformerEncoder
                if hasattr(conformer_enc, 'layers') and len(conformer_enc.layers) > 0:
                    first_layer = conformer_enc.layers[0]
                    layer_gc = getattr(first_layer, 'gradient_checkpointing', None)
                    print(f"     → ConformerEncoder Layer gradient_checkpointing = {layer_gc}")
        
        except Exception as e:
            print(f"❌ Lỗi khi enable: {e}")
            import traceback
            traceback.print_exc()
        
        # 6. Kiểm tra _gradient_checkpointing_func
        print("\n6. Kiểm tra _gradient_checkpointing_func:")
        print("-" * 80)
        
        for name, component in components:
            has_func = hasattr(component, '_gradient_checkpointing_func')
            if has_func:
                func = getattr(component, '_gradient_checkpointing_func', None)
                print(f"✅ {name}: có _gradient_checkpointing_func = {func is not None}")
            else:
                print(f"❌ {name}: KHÔNG có _gradient_checkpointing_func")
        
        # Kiểm tra layers
        if hasattr(model.speech_encoder, 'encoder'):
            conformer_enc = model.speech_encoder.encoder
            if hasattr(conformer_enc, 'layers') and len(conformer_enc.layers) > 0:
                first_layer = conformer_enc.layers[0]
                has_func = hasattr(first_layer, '_gradient_checkpointing_func')
                print(f"{'✅' if has_func else '❌'} ConformerEncoderLayer: có _gradient_checkpointing_func = {has_func}")
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("KẾT THÚC KIỂM TRA")
    print("=" * 80)

if __name__ == "__main__":
    check_gradient_checkpointing()

