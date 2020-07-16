package io.github.jhipster.sample.web.rest;

import io.github.jhipster.sample.prediction.RetinaImageService;
import io.github.jhipster.sample.prediction.RetinalClassificationImpl;
import io.github.jhipster.sample.prediction.RetinalPictureUtils;
import io.github.jhipster.sample.service.BankAccountService;
import io.github.jhipster.sample.web.rest.errors.BadRequestAlertException;
import io.github.jhipster.sample.service.dto.BankAccountDTO;
import io.github.jhipster.sample.service.dto.BankAccountCriteria;
import io.github.jhipster.sample.service.BankAccountQueryService;

import io.github.jhipster.web.util.HeaderUtil;
import io.github.jhipster.web.util.ResponseUtil;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import javax.validation.Valid;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Optional;

/**
 * REST controller for managing {@link io.github.jhipster.sample.domain.BankAccount}.
 */
@RestController
@RequestMapping("/api")
public class BankAccountResource {

    private final Logger log = LoggerFactory.getLogger(BankAccountResource.class);

    private static final String ENTITY_NAME = "testRootBankAccount";

    @Value("${jhipster.clientApp.name}")
    private String applicationName;

    private final BankAccountService bankAccountService;

    private final BankAccountQueryService bankAccountQueryService;

    @Autowired
    ResourceLoader resourceLoader;

    public BankAccountResource(BankAccountService bankAccountService, BankAccountQueryService bankAccountQueryService) {
        this.bankAccountService = bankAccountService;
        this.bankAccountQueryService = bankAccountQueryService;
    }

    /**
     * {@code POST  /bank-accounts} : Create a new bankAccount.
     *
     * @param bankAccountDTO the bankAccountDTO to create.
     * @return the {@link ResponseEntity} with status {@code 201 (Created)} and with body the new bankAccountDTO, or with status {@code 400 (Bad Request)} if the bankAccount has already an ID.
     * @throws URISyntaxException if the Location URI syntax is incorrect.
     */
    @PostMapping("/bank-accounts")
    public ResponseEntity<BankAccountDTO> createBankAccount(@Valid @RequestBody BankAccountDTO bankAccountDTO) throws URISyntaxException {
        log.debug("REST request to save BankAccount : {}", bankAccountDTO);
        if (bankAccountDTO.getId() != null) {
            throw new BadRequestAlertException("A new bankAccount cannot already have an ID", ENTITY_NAME, "idexists");
        }
        BankAccountDTO result = bankAccountService.save(bankAccountDTO);
        return ResponseEntity.created(new URI("/api/bank-accounts/" + result.getId()))
            .headers(HeaderUtil.createEntityCreationAlert(applicationName, true, ENTITY_NAME, result.getId().toString()))
            .body(result);
    }

    /**
     * {@code PUT  /bank-accounts} : Updates an existing bankAccount.
     *
     * @param bankAccountDTO the bankAccountDTO to update.
     * @return the {@link ResponseEntity} with status {@code 200 (OK)} and with body the updated bankAccountDTO,
     * or with status {@code 400 (Bad Request)} if the bankAccountDTO is not valid,
     * or with status {@code 500 (Internal Server Error)} if the bankAccountDTO couldn't be updated.
     * @throws URISyntaxException if the Location URI syntax is incorrect.
     */
    @PutMapping("/bank-accounts")
    public ResponseEntity<BankAccountDTO> updateBankAccount(@Valid @RequestBody BankAccountDTO bankAccountDTO) throws URISyntaxException {
        log.debug("REST request to update BankAccount : {}", bankAccountDTO);
        if (bankAccountDTO.getId() == null) {
            throw new BadRequestAlertException("Invalid id", ENTITY_NAME, "idnull");
        }
        BankAccountDTO result = bankAccountService.save(bankAccountDTO);
        return ResponseEntity.ok()
            .headers(HeaderUtil.createEntityUpdateAlert(applicationName, true, ENTITY_NAME, bankAccountDTO.getId().toString()))
            .body(result);
    }

    /**
     * {@code GET  /bank-accounts} : get all the bankAccounts.
     *
     * @param criteria the criteria which the requested entities should match.
     * @return the {@link ResponseEntity} with status {@code 200 (OK)} and the list of bankAccounts in body.
     */
    @GetMapping("/bank-accounts")
    public ResponseEntity<List<BankAccountDTO>> getAllBankAccounts(BankAccountCriteria criteria) {
        log.debug("REST request to get BankAccounts by criteria: {}", criteria);
        List<BankAccountDTO> entityList = bankAccountQueryService.findByCriteria(criteria);
        return ResponseEntity.ok().body(entityList);
    }

    /**
     * {@code GET  /bank-accounts/count} : count all the bankAccounts.
     *
     * @param criteria the criteria which the requested entities should match.
     * @return the {@link ResponseEntity} with status {@code 200 (OK)} and the count in body.
     */
    @GetMapping("/bank-accounts/count")
    public ResponseEntity<Long> countBankAccounts(BankAccountCriteria criteria) {
        log.debug("REST request to count BankAccounts by criteria: {}", criteria);
        return ResponseEntity.ok().body(bankAccountQueryService.countByCriteria(criteria));
    }

    /**
     * {@code GET  /bank-accounts/:id} : get the "id" bankAccount.
     *
     * @param id the id of the bankAccountDTO to retrieve.
     * @return the {@link ResponseEntity} with status {@code 200 (OK)} and with body the bankAccountDTO, or with status {@code 404 (Not Found)}.
     */
    @GetMapping("/bank-accounts/{id}")
    public ResponseEntity<BankAccountDTO> getBankAccount(@PathVariable Long id) {
        log.debug("REST request to get BankAccount : {}", id);
        Optional<BankAccountDTO> bankAccountDTO = bankAccountService.findOne(id);
        /**
         * TODO start here with analysing
         */

        ByteArrayInputStream bais = new ByteArrayInputStream(bankAccountDTO.get().getAttachment());
        Resource resource = resourceLoader.getResource("");
        try {
            File savedModel = resource.getFile();
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(savedModel);
            BufferedImage retina = ImageIO.read(bais);
            NativeImageLoader loader = new NativeImageLoader(100,400, 3);
            INDArray input = loader.asMatrix(retina);
            ImagePreProcessingScaler pre = new ImagePreProcessingScaler(0,1);
            pre.transform(input);
            INDArray output = model.output(input, false);

            if(output.getFloat(0) > 0.8) {
                bankAccountDTO.get().setRetinaresult("true");
            } else {
                bankAccountDTO.get().setRetinaresult("false");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return ResponseUtil.wrapOrNotFound(bankAccountDTO);
    }

    /**
     * {@code DELETE  /bank-accounts/:id} : delete the "id" bankAccount.
     *
     * @param id the id of the bankAccountDTO to delete.
     * @return the {@link ResponseEntity} with status {@code 204 (NO_CONTENT)}.
     */
    @DeleteMapping("/bank-accounts/{id}")
    public ResponseEntity<Void> deleteBankAccount(@PathVariable Long id) {
        log.debug("REST request to delete BankAccount : {}", id);
        bankAccountService.delete(id);
        return ResponseEntity.noContent().headers(HeaderUtil.createEntityDeletionAlert(applicationName, true, ENTITY_NAME, id.toString())).build();
    }
}
