#!/usr/bin/env python3
import luigi
import pipeline
import tempfile
import sys
import os
from lettucethink.db.fsdb import DB
from shutil import copyfile
from optparse import OptionParser
import json

if __name__ == "__main__":
    usage = "usage: %prog [options] db/scan"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--task",
                      dest="task",
                      default=None)

    parser.add_option("-c", "--config",
                      dest="config",
                      default=None)

    (options, args) = parser.parse_args()

    if '-n' in sys.argv:
        i = find('-n')

    if len(args) != 1:
        raise Exception(
            'Wrong number of arguments. Type %prog --help for more.')

    scan = args[0]
    scan = scan.split('/')
    db = '/'.join(scan[:-1])
    scan = scan[-1]
    print('DB location = %s' % db)
    print('scan = %s' % scan)

    # test = DB(db)
    scan_pipeline = os.path.join(db, scan, "pipeline.toml")

    if options.config is None: # By default, take pipeline.toml in the scan directory
        options.config = scan_pipeline

    if os.path.splitext(options.config)[-1] == ".toml":
        import toml
        config = toml.load(open(options.config))
    elif os.path.splitext(options.config)[-1] == ".json":
        config = json.load(open(options.config))
    else:
        raise Exception('Unknown pipeline config type')

    config['DatabaseConfig'] = {
        'db_location' : db,
        'scan_id' : scan
    }

    # Dirty hack
    with tempfile.NamedTemporaryFile() as tmp:
        tmp = tmp.name
        print(tmp)
        x = open(tmp, 'w')
        for k1 in config.keys():
            x.write('[%s]\n'%k1)
            for k2 in config[k1].keys():
                val = config[k1][k2]
                if isinstance(val, dict):
                    val=json.dumps(val)
                else:
                    val=str(val)
                x.write('%s=%s\n'%(k2,val))
        x.close()
        luigi_config = luigi.configuration.get_config()
        luigi_config.read(tmp)

    # db_config = pipeline.DatabaseConfig(scan_id=scan, db_location=db)
    # cli_args = []
    # config_env = {}

    # print(cli_args)

    luigi.build(tasks=[eval("pipeline.%s()" % options.task)],
                local_scheduler=True)

    toml.dump(config, open(scan_pipeline, 'w'))
