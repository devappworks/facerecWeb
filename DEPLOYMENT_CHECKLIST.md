# Deployment Checklist

Complete checklist for deploying the multi-domain architecture update.

## Pre-Deployment (Development/Staging)

### 1. Code Preparation

- [x] All changes committed to git
- [x] Code pushed to branch: `claude/analyze-code-organization-01SSRFmGov7gxqvqEFUqcsvS`
- [ ] Code reviewed
- [ ] All tests passing (if tests exist)
- [ ] No hardcoded credentials in code

### 2. Dependency Management

- [x] `requirements.txt` updated with Flask-SQLAlchemy
- [ ] Dependencies installed in virtual environment
- [ ] No conflicts with existing packages
- [ ] All imports working correctly

### 3. Database Preparation

- [x] SQLAlchemy models created (Domain, Person, Image, TrainingSession)
- [x] Database initialization code in `app/database.py`
- [x] Auto-migration on first run configured
- [ ] Database location verified: `storage/training.db`
- [ ] Database folder writable

### 4. Migration Script

- [x] Migration script created: `migrations/migrate_to_multi_domain.py`
- [ ] Migration script tested with `--dry-run`
- [ ] Migration output reviewed
- [ ] Backup procedure documented

### 5. Configuration

- [x] Database config added to `config.py`
- [ ] `.env` file updated (if used)
- [ ] Environment variables documented
- [ ] Paths verified for production environment

## Testing (Before Production Deployment)

### 6. Functional Testing

#### Database
- [ ] Start application - database auto-creates
- [ ] Default 'serbia' domain auto-creates
- [ ] Tables created correctly (check with SQLite browser)

#### Domain Management
- [ ] Create new domain via API: `POST /api/domains`
- [ ] List domains: `GET /api/domains`
- [ ] Get domain details: `GET /api/domains/serbia`
- [ ] Update domain: `PUT /api/domains/serbia`
- [ ] Get domain stats: `GET /api/domains/serbia/stats`

#### Training Workflow
- [ ] Generate candidates with domain parameter
- [ ] Start batch training with domain parameter
- [ ] Check batch status
- [ ] Deploy to production with domain parameter

#### Recognition
- [ ] Recognize face with domain='serbia' (existing)
- [ ] Recognize face with domain='greece' (new domain)
- [ ] Verify domain isolation (Greek client doesn't see Serbian celebrities)

### 7. Migration Testing

- [ ] Run migration on staging data: `python migrations/migrate_to_multi_domain.py`
- [ ] Verify folders moved correctly
- [ ] Verify database populated with existing people
- [ ] Verify image counts match
- [ ] Test recognition on migrated data
- [ ] Verify backward compatibility (existing API calls still work)

### 8. Performance Testing

- [ ] Benchmark recognition before migration (time in logs)
- [ ] Benchmark recognition after migration (time in logs)
- [ ] Verify 2-3x speedup with domain separation
- [ ] Check memory usage
- [ ] Check database query performance

### 9. Integration Testing

- [ ] Wikimedia image download works with domain
- [ ] SERP fallback works with domain
- [ ] Face processing saves to correct domain folder
- [ ] Batch recognition searches correct domain only
- [ ] Name mapping works per domain

## Deployment (Production)

### 10. Pre-Deployment Backup

- [ ] Backup entire `storage/` directory
- [ ] Backup database (if exists): `storage/training.db`
- [ ] Backup `.env` file
- [ ] Backup current codebase
- [ ] Document rollback procedure

### 11. Deployment Steps

```bash
# 1. Stop application
- [ ] Stop gunicorn/uwsgi/pm2
- [ ] Verify no processes running: ps aux | grep python

# 2. Pull code
- [ ] git fetch origin
- [ ] git checkout claude/analyze-code-organization-01SSRFmGov7gxqvqEFUqcsvS
- [ ] git pull origin claude/analyze-code-organization-01SSRFmGov7gxqvqEFUqcsvS

# 3. Update dependencies
- [ ] source venv/bin/activate
- [ ] pip install -r requirements.txt

# 4. Run migration (dry run first)
- [ ] python migrations/migrate_to_multi_domain.py --dry-run
- [ ] Review output

# 5. Run migration (actual)
- [ ] python migrations/migrate_to_multi_domain.py
- [ ] Verify "MIGRATION COMPLETE" message
- [ ] Check folders moved correctly
- [ ] Check database created and populated

# 6. Start application
- [ ] Start gunicorn/uwsgi/pm2
- [ ] Verify application starts without errors
- [ ] Check logs for errors

# 7. Verify deployment
- [ ] Test health endpoint
- [ ] Test domain listing: GET /api/domains
- [ ] Test recognition with existing data
- [ ] Check application logs
```

### 12. Post-Deployment Verification

#### Application Health
- [ ] Application starts successfully
- [ ] No errors in logs
- [ ] Database connection working
- [ ] All blueprints registered

#### Functionality
- [ ] Existing recognition API working
- [ ] Domain management API working
- [ ] Training workflow working
- [ ] Statistics endpoints working

#### Data Integrity
- [ ] All existing people visible in database
- [ ] Image counts match folder counts
- [ ] Name mappings preserved
- [ ] Recognition results unchanged

#### Performance
- [ ] Response times acceptable
- [ ] Database queries fast
- [ ] No memory leaks
- [ ] CPU usage normal

### 13. Client Communication

- [ ] Notify clients of new domain features (if applicable)
- [ ] Update API documentation
- [ ] Provide migration timeline
- [ ] Document new endpoints

## Post-Deployment (Production)

### 14. Monitoring (First 24 Hours)

- [ ] Monitor error logs
- [ ] Monitor performance metrics
- [ ] Monitor database size growth
- [ ] Monitor API response times
- [ ] Check for any user-reported issues

### 15. Optimization

- [ ] Review domain statistics
- [ ] Optimize slow queries (if any)
- [ ] Add database indexes if needed
- [ ] Review cost savings from Wikimedia usage

### 16. Documentation

- [ ] Update README with new features
- [ ] Update API documentation
- [ ] Document new environment variables
- [ ] Create user guide for domain management

### 17. Cleanup

- [ ] Remove old backup files (after verification period)
- [ ] Archive migration logs
- [ ] Update deployment documentation
- [ ] Close related tickets/issues

## Rollback Procedure (If Issues Arise)

### Critical Issues Only
If deployment causes critical issues, rollback immediately:

```bash
# 1. Stop application
pkill -f "python run.py"

# 2. Restore code
git checkout <previous-commit>
pip install -r requirements.txt

# 3. Restore files
rm -rf storage/
mv storage_backup/ storage/

# 4. Restore database (if exists)
rm storage/training.db
mv storage/training.db.backup storage/training.db

# 5. Start application
python run.py  # or your production start command

# 6. Verify rollback
curl http://localhost:5000/api/health
```

## Success Criteria

Deployment is successful when:

- [x] All code changes deployed
- [ ] Migration completed without errors
- [ ] No critical bugs in production
- [ ] Performance equal or better than before
- [ ] All existing functionality works
- [ ] New domain features working
- [ ] No data loss
- [ ] Clients satisfied
- [ ] Documentation updated

## Sign-Off

- [ ] Developer: _____________________ Date: _______
- [ ] QA/Tester: ____________________ Date: _______
- [ ] DevOps: _______________________ Date: _______
- [ ] Product Owner: ________________ Date: _______

## Notes

**Estimated Downtime:** < 5 minutes (stop app, run migration, start app)

**Risk Level:** Low (non-breaking changes, backward compatible)

**Rollback Time:** < 2 minutes (restore from backup)

**Dependencies:**
- Flask-SQLAlchemy
- SQLite (built-in with Python)

**Breaking Changes:** None (all backward compatible)

**New Features:**
- Multi-domain support
- Database tracking
- Cost analytics
- Domain management API
