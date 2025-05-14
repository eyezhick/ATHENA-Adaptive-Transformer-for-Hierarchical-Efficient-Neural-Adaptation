"""
Generate ATHENA architecture diagram using Graphviz.
"""

import graphviz

def create_architecture_diagram():
    # Create a new directed graph
    dot = graphviz.Digraph('ATHENA Architecture', format='png')
    dot.attr(rankdir='TB', size='11,8', dpi='300')
    
    # Set node styles
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial')
    dot.attr('edge', fontname='Arial')
    
    # Add main components
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='ATHENA Framework', style='rounded', bgcolor='lightgrey')
        
        # Core Components
        c.node('polyadapter', 'PolyAdapter Layer\n(LoRA + IA³ + MoE)')
        c.node('autorank', 'AutoRank Optimizer')
        c.node('freezing', 'Progressive Freezing\nScheduler')
        c.node('memory', 'Cross-Task Memory')
        
        # Vision-Language Components
        c.node('vl_adapter', 'Vision-Language\nAdapter')
        c.node('qformer', 'Q-Former Adapter')
        
        # Training Components
        c.node('trainer', 'Training Pipeline')
        c.node('evaluator', 'Evaluation Pipeline')
        
        # Model Components
        c.node('base_model', 'Base Model\n(Llama, BLIP-2, etc.)')
        c.node('adapters', 'Parameter-Efficient\nAdapters')
        
        # Add connections
        c.edge('base_model', 'adapters', 'Parameter\nSharing')
        c.edge('adapters', 'polyadapter', 'Feature\nAdaptation')
        c.edge('polyadapter', 'vl_adapter', 'Cross-Modal\nFusion')
        c.edge('vl_adapter', 'qformer', 'Query\nGeneration')
        
        c.edge('autorank', 'adapters', 'Rank\nOptimization')
        c.edge('freezing', 'base_model', 'Layer\nFreezing')
        c.edge('memory', 'adapters', 'Knowledge\nRetrieval')
        
        c.edge('trainer', 'base_model', 'Fine-tuning')
        c.edge('trainer', 'adapters', 'Adapter\nTraining')
        c.edge('evaluator', 'base_model', 'Performance\nEvaluation')
    
    # Add optimization components
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Optimization Techniques', style='rounded', bgcolor='lightgreen')
        
        c.node('deepspeed', 'DeepSpeed\nOptimization')
        c.node('amp', 'Mixed Precision\nTraining')
        c.node('gradient', 'Gradient\nCheckpointing')
        c.node('flash', 'Flash Attention')
        
        c.edge('deepspeed', 'trainer', 'Distributed\nTraining')
        c.edge('amp', 'trainer', 'Memory\nEfficiency')
        c.edge('gradient', 'trainer', 'Memory\nOptimization')
        c.edge('flash', 'trainer', 'Attention\nOptimization')
    
    # Add evaluation components
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Evaluation Metrics', style='rounded', bgcolor='lightyellow')
        
        c.node('metrics', 'Performance\nMetrics')
        c.node('logging', 'Wandb\nLogging')
        
        c.edge('metrics', 'evaluator', 'Model\nAssessment')
        c.edge('logging', 'trainer', 'Training\nMonitoring')
    
    # Save the diagram
    dot.render('docs/architecture', cleanup=True)

if __name__ == '__main__':
    create_architecture_diagram() 