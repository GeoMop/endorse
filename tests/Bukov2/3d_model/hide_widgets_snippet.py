#===========================================
# Manually turn off Show Plane/Clip/Box/...

def hide_widget(item_name):
    item = FindSource(item_name)
    
    renderView1 = GetActiveViewOrCreate('RenderView')
    #prop = GetDisplayProperties(item, view=renderView1)
    #vis = prop.Visibility
    #print(item_name, vis)
    

    if item is None:
        raise AttributeError("Unknown item: ", item_name)
    SetActiveSource(item)
    #item_type = type(item).__name__.split('.')[-1]
    try:
        Show3DWidgets(proxy=item.ClipType)
        Hide3DWidgets(proxy=item.ClipType)
    except AttributeError as e:
        pass
    try:
        Show3DWidgets(proxy=item.SliceType)
        Hide3DWidgets(proxy=item.SliceType)
    except AttributeError as e:
        pass
    Hide(item, renderView1)
    #prop.Visibility = vis
    #renderView1 = GetActiveViewOrCreate('RenderView')
    #prop = GetDisplayProperties(item, view=renderView1)

def hide_all_widgets():
    hide_widget('test_subblock')
    hide_widget('clip-horizontal')

    positions = [2, 4, 6, 8, 10]
    for i in positions:
        hide_widget(f"slice zk30_{i}")
        hide_widget(f"slice zk40_{i}")

hide_all_widgets()
#===========================================

# PUT BEFORE:
# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------
