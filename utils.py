import json
import os
import time
import xml.etree.ElementTree as ET
import requests
import re
import openai
from pymatgen.core import Lattice, Structure,Element
from pymatgen.io.vasp.inputs import Poscar

def edit_job_json(project_id: int, image_address: str = "registry.dp.tech/dptech/vasp:5.4.4", output_file: str = "job.json") -> dict:
    """
    编辑或创建 job.json 文件，设置项目编号和镜像信息
    
    Args:
        project_id (int): 项目编号，例如 21128
        image_address (str): 镜像地址，默认为 "registry.dp.tech/dptech/vasp:5.4.4"
        output_file (str): 输出文件名，默认为 "job.json"
    
    Returns:
        dict: 更新后的配置内容
    """
    # 默认配置
    default_config = {
        "job_name": "Bohrium-VASP",
        "job_type": "container",
        "command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std ",
        "log_file": "tmp_log",
        "backward_files": [],
        "project_id": project_id,
        "platform": "ali",
        "machine_type": "c32_m32_cpu",
        "image_address": image_address
    }
    
    try:
        # 如果文件存在，读取现有配置
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                config = json.load(f)
                # 更新配置
                config["project_id"] = project_id
                config["image_address"] = image_address
        else:
            config = default_config
        
        # 写入文件
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ 成功更新 {output_file}")
        print(f"项目编号: {project_id}")
        print(f"镜像地址: {image_address}")
        
        return config
        
    except Exception as e:
        print(f"❌ 更新 {output_file} 失败: {str(e)}")
        return {}
    

class VasprunParser:
    def __init__(self, path):
        self.path = path

    def get_root(self):
        tree = ET.parse(self.path)
        return tree.getroot()

    def generator(self):
        gen_node = list(self.get_root())[0]
        return [(item.attrib['name'], item.text) for item in gen_node.iter('i')]

    def incar(self):
        inc_node = list(self.get_root())[1]
        return [(item.attrib['name'], item.text) for item in inc_node.iter('i')]

    def monkhorst_pack(self):
        kpoints = list(self.get_root())[2]
        first_block = list(kpoints)[0]
        return [(item.attrib['name'], item.text.split()) for item in list(first_block)]

    def kpoints_list(self):
        kpoints = list(self.get_root())[2]
        coord_block = list(kpoints)[1]
        return [v.text.split() for v in coord_block.findall('v')]

    def kpoints_weight(self):
        kpoints = list(self.get_root())[2]
        children = list(kpoints)
        if len(children) >= 3:
            weight_block = children[2]
            return [v.text.strip() for v in weight_block.findall('v')]
        else:
            return []  # 或者 return ['1.0']*len(self.kpoints_list()) 默认均匀权重

    def parameters(self):
        param = list(self.get_root())[3]
        result = []

        for item in param.iter('i'):
            result.append((item.attrib.get('name'), item.text.strip()))

        for item in param.iter('v'):
            name = item.attrib.get('name')  # 安全访问
            if name is not None:
                result.append((name, item.text.split()))
            else:
                result.append(("unnamed_v", item.text.split()))
        
        return result

    def atoms_info(self):
        atinfo = list(self.get_root())[4]
        children = list(atinfo)
        info = [
            ('No. of atoms', children[0].text.strip()),
            ('atom types', children[1].text.strip())
        ]
        for c in atinfo.iter('c'):
            info.append(c.text.strip())
        return info

    def structure(self):
        cstruct = []
        atom_struct = list(self.get_root())[5]
        crystal = atom_struct.find('crystal')

        if crystal is not None:
            elem_varrays = crystal.findall('varray')
            elemi = crystal.find('i')

            for varray in elem_varrays:
                vectors = [v.text.split() for v in varray.findall('v')]
                cstruct.append((varray.attrib['name'], vectors))

            if elemi is not None:
                cstruct.append((elemi.attrib['name'], elemi.text.strip()))
        else:
            cstruct.append(("crystal", "Not found"))

        for varray in atom_struct.findall('varray'):
            vectors = [v.text.split() for v in varray.findall('v')]
            cstruct.append((varray.attrib['name'], vectors))

        return cstruct

    def calculation(self):
        calc_res = []
        calc_forces = []
        calc_positions = []

        all_steps = list(self.get_root())[6:]  # skip first 6 blocks (header info)

        for step in all_steps:
            tag_list = []
            cpos = []
            cforces = []

            for itag in step.iter('i'):
                tag_list.append((itag.attrib['name'], itag.text.strip()))
            for timetag in step.iter('time'):
                tag_list.append((timetag.attrib['name'], timetag.text.split()))

            structure = step.find('structure')
            if structure is not None:
                varrays = structure.findall('varray')
                if varrays:
                    for v in varrays[0].iter('v'):
                        cpos.append(v.text.split())

            force_varray = step.findall('varray')
            if force_varray:
                for v in force_varray[0].iter('v'):
                    cforces.append(v.text.split())

            calc_res.append(tag_list)
            calc_positions.append(cpos)
            calc_forces.append(cforces)

        return calc_res, calc_positions, calc_forces
    
class SimPleChat:
    
    def __init__(self, system="你是一个在晶体结构领域的专家"):
        self.system =system
        self.refresh()
        

        key_dict = dict(
        DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL"),
        DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY"),
        DEEPSEEK_MODEL_NAME = os.environ.get("DEEPSEEK_MODEL_NAME"),
        )
        
        assert key_dict.get("DEEPSEEK_API_KEY", None) is not None, "Please set the DEEPSEEK_API_KEY in environment."
        assert key_dict.get("DEEPSEEK_BASE_URL", None) is not None, "Please set the DEEPSEEK_BASE_URL in environment."
        self.client = openai.OpenAI(
            api_key=key_dict.get("DEEPSEEK_API_KEY"),
            base_url=key_dict.get("DEEPSEEK_BASE_URL"),
        )
        self.model = key_dict.get("DEEPSEEK_MODEL_NAME","deepseek-chat")

    def refresh(self):
        self.messages = [{"role": "system", "content": self.system}]

    def _ask(self,msg):
        chat_completion = self.client.chat.completions.create(model=self.model,
                                                                messages=msg,
                                                                response_format = {"type": "text"})
        answer = chat_completion.choices[0].message.content
        return answer


    def Q(self,question):
        self.messages.append({"role": "user", "content": str(question)})
        answer = self._ask(self.messages)
        self.messages.append({"role": "assistant", "content": str(answer)})
        return answer
    
    

def make_float(strs):
    if "/" in strs:
        strs = strs.split("/")
        return float(strs[0])/float(strs[1])
    else:
        return float(strs)
    
def map_local_cif(material_id):
    mp_root_dir = os.getenv("MP_ROOT_DIR")
    exact_name = f"{material_id}.cif"
    exact_path = os.path.join(mp_root_dir, exact_name)
    if os.path.isfile(exact_path):
        return Structure.from_file(exact_path)
    else:
        raise FileNotFoundError(f"File {exact_path} not found.")


def get_structure_mp_database(formula,space_group=None):
    from pymatgen.ext.matproj import MPRester
    API_KEY = "msZce01AjFltxEu97whgD2TBdQYwdxhQ"

    # 下载结构并保存为 POSCAR 和 CIF 文件
    with MPRester(API_KEY) as mpr:
        results = mpr.get_entries(formula, sort_by_e_above_hull=True)

    res_data = []
    for n,entry in enumerate(results):
        res = {}
        
        entry_id = entry.entry_id  # e.g., "mp-22474"

        # 获取结构信息
        structure = entry.structure
        
        res["formula"] =  entry.composition.formula
        res["material_id"] =  entry.data.get("material_id","")
        res["entry_id"] =  entry_id
        res["structure"] = structure
        
        if space_group:
            sg,sgn = structure.get_space_group_info()
            if isinstance(space_group, str) and space_group == sg:
                res_data.append(res)
            elif isinstance(space_group, int) and space_group == sgn:
                res_data.append(res)
        else:
            res_data.append(res)
                
        if len(res_data) >= 3:
            break
        
    return res_data
    

def get_structure_dp_database(formula,space_group=None):
    
    url = "https://materials-db-agent.mlops.dp.tech/query/openai"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    if space_group:
        text = f"请给出空间群号为 {space_group} ,化学式为 {formula} ,的晶体结构。"
    else:
        text = f"请给出化学式为 {formula} 的晶体结构。"
    
    payload = {
        "text": text,  # Replace with your actual query text
        "limit": 3        # You can adjust the limit as needed
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()  # Get the JSON response data
        # print("Response data:", data)
    else:
        # print(f"Request failed with status code {response.status_code}")
        # print("Response:", response.text)
        raise Exception(f"Request failed with status code {response.status_code}")
    
    data = data.get("data", [])
    if not data:
        return []
    
    res_data = []
    for item in data:
        res = {}
        res["formula"] =  item.get('formula',"")
        res["material_id"] =  item.get('material_id',"")
        
        # method 1
        structure =  map_local_cif(item.get('material_id'))
        
        res["structure"] = structure
        res_data.append(res)


    return res_data

def rep_string(string):
    if "```json" in string:
        json_pattern = r"```json(.*?)```"
        match = re.search(json_pattern, string, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise ValueError("No JSON block found in the string.")
        string = re.sub(json_pattern, r"\1", code).replace("\n", "")
    else:
        string = string.replace("\n", "")
        string = string.replace("'", '"')
    return string


def check_structure(structure,space_group=None):
    # score = []
    if space_group:
        sg,sgn = structure.get_space_group_info()
        if isinstance(space_group, str) and space_group == sg:
            return True
        elif isinstance(space_group, int) and space_group == sgn:
            return True
        else:
            return False
    else:
        return True
    
def search_poscar_template_tool(formula: str):
    """
    对化学式进行数据库匹配、超胞扩展、原子替换和吸附处理后返回POSCAR模板

    输入：化学式，比如"Sr5Ca3Fe8O24"
    输出：POSCAR模板文件内容，这里只是模板，需要博士生根据模板进行原子替换，确保结构中的所有原子种类、数量和分布都严格符合输入化学式。
    

    """
    # 1. 获得解析后的文献,问询大模型
    ba = SimPleChat(system="你是一个在晶体结构领域的专家")

    prompt1 = """1. 根据下文给出的元素种类及配比,判断是否多类元素通过原子置换的方式占据了同类位点,给出同类位点的元素分组信息。
    如果化学式中,判断部分为掺杂或吸附元素(如H,OH,轻金属等),而不是基础结构本身元素,分别在base、adsorb分别给出基础结构和掺杂/吸附的元素配比信息。adsorb元素不需要进行分组。
    例如 "Sr3CaTi4O12"中, 假定Sr与Ca占据同类位点,Sr3CaTi4O12的那么化学式模板为A4B4O12,根据公约数,可简化为 ABO3 形式,而该形式正好符合钙钛矿ABO3模板。
    因此 根据晶体学经验,"Sr3CaTi4O12" 的分组信息如下：
    {
    "base":{
        "A":{"Sr":3,"Ca":1,"all":4},
        "B":{"Ti":4,"all":4},
        "C":{"O":12,"all":12},
    },
    "adsorb":{}
    }
    注意,上述分析应当首先检测是否已经符合已知材料种类配比,例如"Li7La3Zr2O12",为已知材料类型,分组信息可以为
    {
    "base":{
            "A":{"Li":7,"all":7},
            "B":{"La":3,"all":3},
            "C":{"Zr":2,"all":2},
            "D":{"O":12,"all":12},
            },
    "adsorb":{}
    }
    若原始化学式或分组后的化学式均无法符合已知材料配比，则每种元素均为单独位点。
    """
    res = ba.Q(prompt1)
    # print(res)
    prompt2 = f"需要分析的化学式为 {formula}"
    res = ba.Q(prompt2)
    # print(res)
    prompt3 = """
    将上述结果,按照各位点(A,B,C,...)配比的公约数尝试简化为最简整数配比,并分别将同位置的元素类别分别带入位点,获得多个不同的化学式及公约数。
    例如"SrCaTi2O6"的A的配比和(all)为4, B的配比和(all)为4, C的配比和(all)为12, A:B:C位点配比为4:4:12,可以化简为1:1:3的最简整数配比ABC3,公约数为4。
    注意不考虑元素的配比!!!仅分别将元素类别代入最简整数配比模板ABC3,获得化学式为"SrTiO3"与"CaTiO3"。
    返回所有可能的化学式(包含减去吸附原子的base化学式)的字典及公约数, 如下：
    
    {
    "base":{
        "A":{"Sr":3,"Ca":1,"all":4},
        "B":{"Ti":4,"all":4},
        "C":{"O":12,"all":12},
    },
    "adsorb":{},
    "temps":{"Sr3CaTi4O12":1,"SrTiO3":4,"CaTiO3":4}
    }
    
    掺杂元素adsorb, 直接给出元素及配比信息。
    注意：只返回结果字典,不要任何其他解释性信息。
    
    """
    res = ba.Q(prompt3)
    # print(res)
    prompt4 = "重新上述过程,检查两次结果一致性,只返回可以直接被json解析的结果字典,不要任何其他解释性信息。"
    res  = ba.Q(prompt4)
    res = rep_string(res)
    res = json.loads(res)
    
    base= res.get("base",{})
    ri= res.get("temps",{})
    adsorb = res.get("adsorb",{})

    
    msgs = []

    for k,v in ri.items():
        formula = k # 化学式
        v = int(v) # 超胞大小
        sup = {1:(1,1,1),2:(2,1,1),3:(3,1,1),4:(2,2,1),5:(5,1,1),6:(2,3,1),
                7:(7,1,1),8:(2,2,2),9:(3,3,1),10:(5,2,1),11:(11,1,1),12:(3,2,2),}
        msg = get_structure_dp_database(formula,space_group=None)
        
        if len(msg) == 0:
            pass
        else:
            for mi in msg:
                structure:Structure = mi.get("structure")
                structure2 = structure.copy()
                structure2 = structure2.make_supercell(sup.get(v))
                mi["structure"] = structure2
                mi["space_group"] = structure2.get_space_group_info()
                
                if v==1:
                    pass
                else:
                    for bk,bv in base.items():
                        if len(bv)==2:
                            bv.pop("all",None)
                            pass
                        else:
                            bv.pop("all",None)
                            names = list(bv.keys())
                            namess = []
                            for name,number in bv.items():
                                if isinstance(number, float):
                                    warnings.warn(f"Warning: Can't deal with float ratios {bv}, the result could be wrong.")

                                namess.extend([name]*int(number))
                                
                            names_all = structure2.symbol_set
                            
                            indexes = [i for i, name in enumerate(names_all) if name in names]
                            
                            if len(indexes) != len(namess):
                                pass
                            else:
                                for i in range(len(indexes)):
                                    structure2.replace(indexes[i], namess[i])
                
                if len(adsorb) != 0:
                    prompt5 = "吸附/掺杂元素种类及个数信息为："
                    prompt5 += str(adsorb)
                    prompt5 += f"请根据已有基底结构{structure2}, 估计合理的每个吸附/掺杂物原子的分数x,y,z坐标信息，各原子之间键长合理。"
                    prompt5 += """例如 {"O":2,"H":1} 的返回格式如:
                    {
                    "O": [[0.5, 0.22, 0.48], [0.5, 0.23, 0.5]], 
                    "H": [[0.5, 0.23, 0.52]]
                    }
                    """
                    prompt5 += "注意：只返回吸附/掺杂位点的结果字典,不要任何其他解释性信息。"
                    res = ba.Q(prompt5)
                    # print(res)
                    res = rep_string(res)
                    res = json.loads(res)
                    
                    res_add = {}
                    for resk,resv in res.items():
                        check = Element.is_valid_symbol(resk)
                            
                        if check and isinstance(resv, list):
                            for resvi in resv:
                                res_add[resk] = resvi
                else:
                    res_add = {}
                
                if len(res_add)>0:
                    for res_kk,res_vv in res_add.items():
                        structure2=structure2.append(res_kk,res_vv)
                        
                mi["adsorb"] = adsorb
                mi["base"] = base
                mi["species"] = structure2.symbol_set
                mi["abc"] = structure2.lattice.abc
                mi["angle"] = structure2.lattice.angles
                mi["fractional_coords"] = structure2.frac_coords.tolist()
                mi["reference_formula"] = mi.pop("formula")
                mi["reference_material_id"] = mi.pop("material_id")
                mi["poscar"] = Poscar(structure2)
                mi["poscar_str"] = str(mi["poscar"])
                
                msgs.append(mi)

                
    if len(msgs) == 0:
        raise ValueError("未能获取合适的结构模板。")
    
    with open('prompt/format.txt', 'r') as file:
        vt_format = file.read()
    return {"poscar_template":msgs[0]["poscar_str"], "vt_format":vt_format}



async def ask_human_for_advice(question: str):
    """
    "You can ask a human for guidance when you think you "
    "got stuck or you are not sure what to do next. "

    Returns:
        human's advice
    """
    print("🤖: ", question)
    print("🧑: ")
    response = input().lower()

    return response

async def show_task_status(status: str):
    """
    show the task status for user

    input : what you need to do next
    return : None
    """
    print("🤖: \n", status)
    print("🧑: ")
    return None