
export const app = window.comfyAPI.app.app;
export const api = window.comfyAPI.api.api;
export const $el = window.comfyAPI.ui.$el;

// document.head.insertAdjacentHTML('beforeend', '<link rel="stylesheet" href="extensions/comfyui_controlnet_aux/index.css">');
const style = document.createElement("style"); document.head.append(style); 

app.registerExtension({
    name: 'comfy.ControlNet Preprocessors.Preprocessor Selector',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if(nodeData.name == 'ControlNetPreprocessorSelector'){ const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() { 
                this.setProperty("values",[]); this.setSize([300, 350]); let preprolist=[]; let controlnet='';

                function addhide() { let searchvalue = searchdom.value;
                    selectorlist[1].querySelectorAll(".Preprocessor-tag").forEach( el => el.classList.toggle('hide', !( 
                        el.dataset.tag.toLowerCase().includes(searchvalue) || el.children[0].checked ) ) ) };

                function TagList() { const taglist = preprolist.map(preproname => $el('label.Preprocessor-tag', {
                    dataset: { tag: preproname }, onclick: () => updatalist(selectorlist[1], [preproname])
                    }, [ $el("input", { type: 'checkbox' }), $el("span", { textContent: preproname }) ] ) );

                    taglist.push($el('h6', { textContent: '工作流菜单--浏览模板--自定义节点--comfyui_controlnet_aux--示例工作流' }));

                    return taglist; };                        
                
                function updatalist(els,preproname) { preprolist.sort((a,b)=> preproname.includes(b) - preproname.includes(a));
                    els.innerHTML = ''; els.append(...TagList()); els.children[0].children[0].setAttribute('checked', 'checked'); addhide() };   
                
                async function getprepro(){ const resp = await api.fetchApi(`/Preprocessor?name=${controlnet}`);
                    preprolist = await resp.json(); let mlist = ["","none"];
                    if(controlnet.includes("canny")){ mlist=["canny", "CannyEdgePreprocessor"] }
                    else if(controlnet.includes("depth")){ mlist=["depth", "DepthAnythingV2Preprocessor"] }
                    else if(controlnet.includes("lineart")){ mlist=["lineart", "LineartStandardPreprocessor"] }
                    else if(controlnet.includes("tile")){ mlist=["tile", "TilePreprocessor"] }
                    else if(controlnet.includes("scrib")){ mlist=["scrib", "Scribble_XDoG_Preprocessor"] }
                    else if(controlnet.includes("soft")){ mlist=["soft", "HEDPreprocessor"] }
                    else if(controlnet.includes("pose")){ mlist=["pose", "DWPreprocessor"] }    
                    else if(controlnet.includes("normal")){ mlist=["normal", "BAE-NormalMapPreprocessor"] }
                    else if(controlnet.includes("seg")){ mlist=["semseg", "OneFormer-COCO-SemSegPreprocessor"] }
                    else if(controlnet.includes("shuffle")){ mlist=["shuffle", "ShufflePreprocessor"] }
                    else if(controlnet.includes("ioclab_sd15_recolor")){ mlist=["image", "ImageLuminanceDetector"] }
                    searchdom.value = mlist[0]; updatalist(selectorlist[1],[mlist[1]]); };
                
                const toolsElement = $el('div.tools', [         
                    $el('button.Empty',{ textContent: 'Empty', onclick:()=>{ searchdom.value = ''; addhide() } }),                            
                    $el('textarea.searchpre',{ placeholder:"🔎 searchpre", oninput:(e)=> addhide() }) ]);  

                let selector = this.addDOMWidget("select_styles", "btn", $el('div.Preprocessor', [toolsElement, $el("ul.Preprocessor-list", [])]), {
                    setValue(value) { setTimeout(_=> updatalist(selectorlist[1],[value]), 333); }, getValue: () => preprolist[0] } );
                    
                let selectorlist = selector.element.children; let searchdom = selectorlist[0].querySelector('.searchpre');
                
                Object.defineProperty( this.widgets.find(w => w.name === 'cn'), 'value', { 
                    get:()=> controlnet, set:(value)=>{ controlnet = value; getprepro() } }); 
                
                return onNodeCreated;
            }
        }
    }
})



style.textContent = `
.Preprocessor .tools {
    display: flex;
    justify-content: space-between;
    height: 20px;
    padding-bottom: 5px;
    border-bottom: 2px solid var(--border-color);
}
.Preprocessor .tools button.Empty {
    height: 20px;
    border-radius: 8px;
    border: 2px solid var(--border-color);
    font-size: 11px;
    background: var(--comfy-input-bg);
    color: var(--error-text);
    cursor: pointer;
}
.Preprocessor .tools textarea.searchpre {
    flex: 1;
    margin-left: 10px;
    height: 20px;
    line-height:8px;
    border-radius: 8px;
    border: 2px solid var(--border-color);
    font-size: 15px;
    background: var(--comfy-input-bg);
    color: var(--input-text);
    padding: 4px 10px;
    outline: none;
    resize: none;
}
.Preprocessor-list {
    list-style: none;
    padding: 0;
    margin: 0;
    min-height: 150px;
    height: calc(100% - 30px);
    overflow: auto;
}
.Preprocessor-tag {
    display: inline-block;
    vertical-align: middle;
    margin-right: 0px;
    padding: 0px;
    color: var(--input-text);
    background-color: var(--comfy-input-bg);
    border-radius: 8px;
    border: 2px solid var(--border-color);
    font-size: 11px;
    cursor: pointer;
}
.Preprocessor-tag.hide {
    display: none;
}
.Preprocessor-tag:hover {
    filter: brightness(2);
}
  `;