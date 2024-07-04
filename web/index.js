
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

const link = document.createElement("link"); link.rel = "stylesheet";
link.href = "extensions/comfyui_controlnet_aux_pre/index.css";
document.head.appendChild(link);

let preprolist = {}; let controlnet = 'control_v11p_sd15_canny.pth';

function listsort(listdata,valuestr) { listdata.sort((a,b)=> valuestr.includes(b.name) - valuestr.includes(a.name)); };

function addhide(el,searchValue) { el.classList.toggle('hide', !( 
    el.dataset.name.toLowerCase().includes(searchValue.toLowerCase()) ||  
    el.dataset.tag.toLowerCase().includes(searchValue.toLowerCase()) ||  
    el.classList.contains('Preprocessor-tag-selected') ) ) };
  
function getTagList() {  return preprolist[controlnet].map((tag, index) => { 
    return $el('label.Preprocessor-tag',   
                { dataset: { tag: tag.name, name: tag.name, index: index },  
                  $: (el) => { el.firstChild.onclick = () => { el.classList.toggle("Preprocessor-tag-selected"); }; },  },   
                [ $el("input", { type: 'checkbox', name: tag.name }), $el("span", { textContent: tag.name })  ] ); } );  };

async function getprepro(el){ const resp = await api.fetchApi(`/Preprocessor?name=${controlnet}`);
    if (resp.status === 200) { let data = await resp.json(); let mlist = ["","none"];

        if(controlnet.includes("canny")){ mlist=["canny", "CannyEdgePreprocessor"] }
        else if(controlnet.includes("depth")){ mlist=["depth", "MiDaS-DepthMapPreprocessor"] }
        else if(controlnet.includes("lineart")){ mlist=["lineart", "LineArtPreprocessor"] }
        else if(controlnet.includes("tile")){ mlist=["tile", "TilePreprocessor"] }
        else if(controlnet.includes("scrib")){ mlist=["scrib", "FakeScribblePreprocessor"] }
        else if(controlnet.includes("soft")){ mlist=["soft", "HEDPreprocessor"] }
        else if(controlnet.includes("pose")){ mlist=["pose", "DWPreprocessor"] }    
        else if(controlnet.includes("normal")){ mlist=["normal", "BAE-NormalMapPreprocessor"] }
        else if(controlnet.includes("semseg")){ mlist=["semseg", "OneFormer-ADE20K-SemSegPreprocessor"] }
        else if(controlnet.includes("shuffle")){ mlist=["shuffle", "ShufflePreprocessor"] }
        else if(controlnet.includes("ioclab_sd15_recolor")){ mlist=["image", "ImageLuminanceDetector"] }
        else if(controlnet.includes("t2iadapter_color")){ mlist=["color", "ColorPreprocessor"] }
        else if(controlnet.includes("sketch")){ mlist=["scrib", "FakeScribblePreprocessor"] }
        
        document.querySelector('.search').value = mlist[0]; listsort(data,mlist[1]);

        preprolist[controlnet] = data; el.innerHTML = ''; el.append(...getTagList()); 
        el.children[0].classList.add("Preprocessor-tag-selected"); el.children[0].children[0].checked = true;
    } };

app.registerExtension({
    name: 'comfy.ControlNet Preprocessors.Preprocessor Selector',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData.name == 'ControlNetPreprocessorSelector'){ const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() { const styles_id = this.widgets.findIndex((w) => w.name == 'cn'); 
                this.setProperty("values",[]); this.setSize([300, 350]); 
                
                const toolsElement = $el('div.tools', [         //添加清空按钮搜索框 
                    $el('button.delete',{ textContent: 'Empty',
                        onclick:()=>{ selectorlist[0].querySelectorAll(".search").forEach(el=>{ el.value = '' });
                                      selectorlist[1].querySelectorAll(".Preprocessor-tag").forEach(el => {
                                          el.classList.remove("Preprocessor-tag-selected"); 
                                          el.classList.remove("hide"); el.children[0].checked = false }) } }),
                            
                    $el('textarea.search',{ placeholder:"🔎 search",
                        oninput:(e)=>{ let searchValue = e.target.value;
                            selectorlist[1].querySelectorAll(".Preprocessor-tag").forEach(el => {  addhide(el,searchValue); }) } }) 
                ]);                
                const stylesList = $el("ul.Preprocessor-list", []);               
                let selector = this.addDOMWidget( 'select_styles', "btn", $el('div.Preprocessor', [toolsElement, stylesList] ) ); 
                let selectorlist = selector.element.children;

                // 监听鼠标离开事件  
                selectorlist[1].addEventListener('mouseleave', function(e) { const searchValue = document.querySelector('.search').value;  
                    const selectedTags = Array.from(this.querySelectorAll('.Preprocessor-tag-selected')).map(el => el.dataset.tag); // 当前选中的标签值
                    listsort(preprolist[controlnet],selectedTags); this.innerHTML = ''; this.append(...getTagList());  // 重新排序
                    this.querySelectorAll('.Preprocessor-tag').forEach(el => {  // 遍历所有标签
                        const isSelected = selectedTags.includes(el.dataset.tag); //标签的选中状态
                        if (isSelected) { el.classList.add("Preprocessor-tag-selected"); el.children[0].checked = true; } //更新样式标签的选中状态 
                        addhide(el,searchValue); });   // 同时处理搜索和隐藏逻辑 
                });            
                 
                //根据controlnet模型返回预处理器列表
                Object.defineProperty( this.widgets[styles_id], 'value', { get:()=>{ return controlnet },
                                                                           set:(value)=>{ controlnet = value; getprepro(selectorlist[1]) } })     

                //根据选中状态返回预处理器
                let style_select_values = ''  
                Object.defineProperty(selector, "value", {
                    set: (value) => {           
                        selectorlist[1].querySelectorAll(".Preprocessor-tag").forEach(el => {
                            if (value.split(',').includes(el.dataset.tag)) {
                                el.classList.add("Preprocessor-tag-selected"); el.children[0].checked = true } }) },
                    get: () => {
                        selectorlist[1].querySelectorAll(".Preprocessor-tag").forEach(el => {
                            if(el.classList.value.indexOf("Preprocessor-tag-selected")>=0){
                                if(!this.properties["values"].includes(el.dataset.tag)){ 
                                    this.properties["values"].push(el.dataset.tag); }}
                            else{ if(this.properties["values"].includes(el.dataset.tag)){ 
                                     this.properties["values"]= this.properties["values"].filter(v=>v!=el.dataset.tag); } } });
                        style_select_values = this.properties["values"].join(',');
                        return style_select_values; } });

                getprepro(selectorlist[1]); return onNodeCreated;
            }
        }
    }
})