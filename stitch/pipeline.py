import luigi
from lettucethink.db.fsdb import DB
import match_images as mi
import json
import stitch

IMAGES_DIRECTORY = "images"

class DatabaseConfig(luigi.Config):
    db_location = luigi.Parameter()
    scan_id = luigi.Parameter()


class FilesetTarget(luigi.Target):
    def __init__(self, db_location, scan_id, fileset_id):
        db = DB(db_location)
        db.connect()
        scan = db.get_scan(scan_id)
        if scan is None:
            raise Exception("Scan does not exist")
        self.scan = scan
        self.fileset_id = fileset_id

    def create(self):
        return self.scan.create_fileset(self.fileset_id)

    def exists(self):
        fs = self.scan.get_fileset(self.fileset_id)
        return fs is not None and len(fs.get_files()) > 0

    def get(self, create=True):
        return self.scan.get_fileset(self.fileset_id, create=create)

class RomiTask(luigi.Task):
    def output(self):
        fileset_id = self.task_id
        return FilesetTarget(DatabaseConfig().db_location, DatabaseConfig().scan_id, fileset_id)

    def complete(self):
        outs = self.output()
        if isinstance(outs, dict):
            outs = [outs[k] for k in outs.keys()]
        elif isinstance(outs, list):
            pass
        else:
            outs = [outs]

        if not all(map(lambda output: output.exists(), outs)):
            return False

        req = self.requires()
        if isinstance(req, dict):
            req = [req[k] for k in req.keys()]
        elif isinstance(req, list):
            pass
        else:
            req = [req]
        for task in req:
            if not task.complete():
                return False
        return True


class StitchedMap(RomiTask):
	
    def requires(self):
       return MatchedImages()

    def run(self):
       input_fileset = FilesetTarget(
                        DatabaseConfig().db_location, DatabaseConfig().scan_id, 
                        IMAGES_DIRECTORY).get()
       files=input_fileset.get_files()     
       ids=[f.get_id() for f in files]
       files=[f for _,f in sorted(zip(ids,files))]
  
       homs_file=self.input().get().get_file("homographies").read_text()	
       homs=json.loads(homs_file)["homographies"]
        
       stitched_map=stitch.map(files, homs)
       output_fileset = self.output().get()
       f = output_fileset.get_file('stitched_map', create=True) 
       f.write_image("png", stitched_map)

class MatchedImages(RomiTask):

  def requires(self):
     return []

  def run(self):
     input_fileset = FilesetTarget(
                        DatabaseConfig().db_location, DatabaseConfig().scan_id, 
                        IMAGES_DIRECTORY).get()
     
     filter_params={'x1'   : 0,
                       'x2'   : 10000,
                       'ymin' : 0,
                       'xmin1' :0,
                       'xmax1' :10000,
                       'xmin2' :0,
                       'xmax2' : 10000,
                       'dxmin': 0,
                       'dymin': 0,
                       'dxmax': 10000,
                       'dymax': 10000,
                       'match_ratio':.5
             }     
     files=input_fileset.get_files()#[::-1]     
     ids=[f.get_id() for f in files]
     files=[f for _,f in sorted(zip(ids,files))]

     output_fileset = self.output().get()

     homs={}
     for i in range(len(files)-1):
        print("Matching %s and %s"%(files[i].get_id(),files[i+1].get_id()))
        homs["%s_%s"%(files[i].get_id(),files[i+1].get_id())]=\
        	               {"source" : files[i].get_id(),
                          "target"  : files[i+1].get_id(),
                          "homography": stitch.estim(files[i].read_image(), files[i+1].read_image(), filter_params).tolist()
        	                 }
     f = output_fileset.get_file('homographies', create=True) 
     f.write_text('json', json.dumps({"homographies": homs}, indent=4))