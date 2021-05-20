# jonaslab-cpp

Here are some helpful cellprofiler plugins used by the Laboratory for Bio-Micro Devices at Brigham & Women's Hospital.

## Getting Started

### Prerequisites

[Download CellProfiler](https://cellprofiler.org/releases/). Only CellProfiler 3 is supported for now.
 
### Installing

Download the zip file and move the module files in <code>plugins_py2</code> to your plugins folder (if it exists), or point your CellProfiler's plugin folder in the preferences dialog to your plugins folder.

Or, use <code>git clone</code> in an empty folder.

## Development

### Type hints

Type annotations aid the developer in enforcing variable and argument types. Modern IDEs are able to infer types, but this may not always be the intended types. They are contracts between developers to eliminate ambiguity.

Unfortunately, the version of python shipped with CellProfiler does not support type annotations (yet), so we use the <code>strip-hints</code> module to help us remove type annotations. These <code>Modules</code> can then be used directly in CellProfiler. Run <code>remove_type_hints.py</code> to do so.

### Some notes on modules

Each custom module has to subclass <code>cellprofiler.module.Module</code>. Three class attributes are required:
1. <code>module_name: str</code>, the module name
2. <code>category: Union\[str, List\[str\]\]</code>, the category or categories the module can be classified into
3. <code>variable_revision_number: int</code>, the version number

### Settings

Each module contains one or more <code>cellprofiler.setting.Setting</code>s, which are essentially key-value pairs. Settings can be stored anywhere within the <code>module</code>, as long as the following API holds:
1. <code>create_settings -> None</code> initializes settings and hooks up behavior with button.
2. <code>settings -> List\[Setting\]</code> specifies the order in which settings are serialized to and from the <code>.cpproj</code> file.
3. <code>visible_settings -> List\[Setting\]</code> specifies the list of visible settings.
4. <code>help_settings -> List\[Setting\]</code> specifies the list of help buttons for the settings.


### Measurements

#### Before

Each module may require measurements made by a module earlier in the pipeline. These can be accessed through <code>workspace.measurements.get_measurement(object_name: str, feature_name: str, image_set_number: Union\[int, None\] = None)</code>. 

Feature names are strings separated by underscores, but are accessed in the user interface like a tree (e.g. in the ExportMeasurements module). 

#### After

Each module makes one or more measurements, which are instances of <code>HDF5Dict</code>. The following API exists:
1. <code>get_measurement_columns -> List\[Tuple\[str, str, str\]\]</code> specifies the measurements that will be made at the end of the module
